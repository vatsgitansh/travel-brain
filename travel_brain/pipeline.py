"""
pipeline.py — Master orchestration script for the Travel Brain data pipeline.

Full flow: Scrape → Clean → Chunk → Embed → Tag Metadata → Upsert to VectorDB

Usage examples:
  python travel_brain/pipeline.py                              # All locations, all sources
  python travel_brain/pipeline.py --location bali             # Bali only
  python travel_brain/pipeline.py --location dubai --source reddit
  python travel_brain/pipeline.py --location bali --dry-run   # No DB writes

Environment:
  VECTOR_DB_PROVIDER=chroma   (local, default)
  VECTOR_DB_PROVIDER=pinecone (production)
  EMBEDDING_PROVIDER=local    (local sentence-transformers, default)
  EMBEDDING_PROVIDER=openai   (OpenAI API)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# ── Setup path ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from travel_brain import config

# ── Scrapers ──────────────────────────────────────────────────────────────────
from travel_brain.scrapers.youtube_scraper import scrape_youtube, save_raw as yt_save
from travel_brain.scrapers.reddit_scraper  import scrape_reddit,  save_raw as rd_save
from travel_brain.scrapers.blog_scraper    import scrape_blogs,   save_raw as blog_save

# ── Processing ────────────────────────────────────────────────────────────────
from travel_brain.processing.cleaner          import clean_documents
from travel_brain.processing.chunker          import chunk_documents
from travel_brain.processing.embedder         import embed_chunks
from travel_brain.processing.metadata_builder import build_metadata_batch

# ── Vector DB ─────────────────────────────────────────────────────────────────
from travel_brain.vectordb.base_client import VectorDBClient

console = Console()

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.DATA_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")


# ── VectorDB Factory ──────────────────────────────────────────────────────────

def get_vector_db() -> VectorDBClient:
    provider = config.VECTOR_DB_PROVIDER
    if provider == "pinecone":
        from travel_brain.vectordb.pinecone_client import PineconeClient
        return PineconeClient()
    else:
        from travel_brain.vectordb.chroma_client import ChromaClient
        return ChromaClient()


# ── Source Entry Type ─────────────────────────────────────────────────────────

class SourceEntry(TypedDict):
    fn:    Callable[[str, int], list[dict]]
    save:  Callable[[str, list[dict]], Path]
    label: str


SOURCES: dict[str, SourceEntry] = {
    "youtube": {"fn": scrape_youtube, "save": yt_save,   "label": "YouTube Transcripts"},
    "reddit":  {"fn": scrape_reddit,  "save": rd_save,   "label": "Reddit Posts"},
    "blog":    {"fn": scrape_blogs,   "save": blog_save, "label": "Niche Blogs"},
}


# ── Pipeline Step: Scrape ─────────────────────────────────────────────────────

def run_scrape(location: str, source: str, limit: int, dry_run: bool) -> list[dict]:
    entry: SourceEntry = SOURCES[source]
    label  = entry["label"]
    console.print(f"  🔍 Scraping [bold cyan]{label}[/] for [bold yellow]{location.title()}[/]...")

    scrape_fn: Callable[[str, int], list[dict]] = entry["fn"]
    raw_docs = scrape_fn(location, limit)
    console.print(f"     ↳ {len(raw_docs)} raw documents collected")

    if raw_docs and not dry_run:
        save_fn: Callable[[str, list[dict]], Path] = entry["save"]
        saved = save_fn(location, raw_docs)
        console.print(f"     ↳ Raw data saved → {saved.name}")

    return raw_docs


# ── Pipeline Step: Process ────────────────────────────────────────────────────

def run_process(raw_docs: list[dict]) -> list[dict]:
    console.print("  🧹 Cleaning & deduplicating...")
    cleaned = clean_documents(raw_docs)
    console.print(f"     ↳ {len(cleaned)} documents after cleaning")

    console.print("  ✂️  Chunking...")
    chunks = chunk_documents(cleaned)
    console.print(f"     ↳ {len(chunks)} chunks created")

    console.print("  🏷️  Tagging metadata...")
    tagged = build_metadata_batch(chunks)

    return tagged


# ── Pipeline Step: Embed & Upsert ─────────────────────────────────────────────

def run_embed_and_upsert(
    chunks: list[dict],
    location: str,
    db: VectorDBClient,
    dry_run: bool,
) -> int:
    console.print(f"  🔢 Generating embeddings via [bold]{config.EMBEDDING_PROVIDER}[/]...")
    embedded = embed_chunks(chunks)
    console.print(f"     ↳ {len(embedded)} embeddings generated (dim={config.EMBEDDING_DIM})")

    if dry_run:
        console.print("  [yellow]⚠ DRY RUN — skipping vector DB upsert[/]")
        # Save processed output locally for inspection
        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = config.PROCESSED_DIR / f"chunks_{location}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(
                [{k: v for k, v in c.items() if k != "embedding"} for c in embedded],
                f, indent=2, ensure_ascii=False,
            )
        console.print(f"     ↳ Processed chunks saved → {out_path.name}")
        return 0

    console.print(f"  💾 Upserting to [bold]{config.VECTOR_DB_PROVIDER}[/] (namespace='{location}')...")
    upserted = db.upsert(embedded, namespace=location)
    console.print(f"     ↳ {upserted} vectors upserted")
    return upserted


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    locations: list[str],
    sources: list[str],
    limit: int,
    dry_run: bool,
) -> dict:
    """Run the full pipeline and return a summary stats dict."""
    start = time.time()

    console.print(Panel(
        f"[bold]🧠 Travel Brain — Data Ingestion Pipeline[/]\n"
        f"Locations : {', '.join(l.title() for l in locations)}\n"
        f"Sources   : {', '.join(sources)}\n"
        f"Limit     : {limit} docs/source\n"
        f"Embedder  : {config.EMBEDDING_PROVIDER}\n"
        f"Vector DB : {config.VECTOR_DB_PROVIDER}\n"
        f"Dry Run   : {'YES' if dry_run else 'NO'}",
        title="Pipeline Config",
        border_style="bright_blue",
    ))

    db: Optional[VectorDBClient] = None if dry_run else get_vector_db()

    locations_data: dict[str, Any] = {}
    total_chunks:   int = 0
    total_vectors:  int = 0
    stats: dict[str, Any] = {"locations": locations_data, "total_vectors_upserted": 0, "total_chunks": 0}

    for location in locations:
        console.rule(f"[bold green]{location.upper()}[/]")
        # Explicit local counters so type checker can track int types correctly
        loc_chunks:  int = 0
        loc_vectors: int = 0
        loc_sources: dict[str, dict[str, int]] = {}

        for source in sources:
            console.print(f"\n[bold]{source.upper()}[/]")
            try:
                raw_docs = run_scrape(location, source, limit, dry_run)
                if not raw_docs:
                    console.print("  [yellow]No documents collected — skipping[/]")
                    continue

                chunks  = run_process(raw_docs)
                vectors = run_embed_and_upsert(chunks, location, db, dry_run)

                loc_sources[source] = {
                    "raw_docs": len(raw_docs),
                    "chunks":   len(chunks),
                    "vectors":  vectors,
                }
                loc_chunks  += len(chunks)
                loc_vectors += vectors

            except Exception as e:
                logger.error(f"Pipeline failed for {location}/{source}: {e}", exc_info=True)
                console.print(f"  [bold red]ERROR: {e}[/]")

        loc_stats: dict[str, Any] = {
            "sources": loc_sources,
            "chunks":  loc_chunks,
            "vectors": loc_vectors,
        }
        stats["locations"][location] = loc_stats
        total_chunks  += loc_chunks
        total_vectors += loc_vectors

    elapsed = time.time() - start
    elapsed_rounded: float = round(elapsed, 1)
    stats["elapsed_seconds"] = elapsed_rounded
    stats["total_chunks"] = total_chunks
    stats["total_vectors_upserted"] = total_vectors

    # ── Print Summary Table ───────────────────────────────────────────────────
    table = Table(title="Pipeline Summary", border_style="green")
    table.add_column("Location",  style="cyan")
    table.add_column("Source",    style="white")
    table.add_column("Raw Docs",  justify="right")
    table.add_column("Chunks",    justify="right")
    table.add_column("Vectors",   justify="right", style="green")

    for loc, ls in stats["locations"].items():
        for src, ss in ls["sources"].items():
            table.add_row(loc.title(), src, str(ss["raw_docs"]), str(ss["chunks"]), str(ss["vectors"]))

    console.print("\n")
    console.print(table)
    console.print(f"\n⏱ Completed in {elapsed:.1f}s | "
                  f"Total chunks: {stats['total_chunks']} | "
                  f"Vectors upserted: {stats['total_vectors_upserted']}")

    if db and not dry_run:
        try:
            db_stats = db.describe()
            console.print(f"📊 Vector DB stats: {db_stats}")
        except Exception:
            pass

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Travel Brain — Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python travel_brain/pipeline.py
  python travel_brain/pipeline.py --location bali --source reddit --dry-run
  python travel_brain/pipeline.py --location dubai --source youtube
  python travel_brain/pipeline.py --location bali --limit 20
        """,
    )
    parser.add_argument(
        "--location", choices=[*config.LOCATIONS, "all"], default="all",
        help="Location to scrape (default: all)"
    )
    parser.add_argument(
        "--source", choices=[*SOURCES.keys(), "all"], default="all",
        help="Data source to use (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=config.SCRAPE_LIMIT,
        help=f"Max documents per source (default: {config.SCRAPE_LIMIT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing to vector DB. Saves processed output to data/processed/"
    )
    args = parser.parse_args()

    locations = config.LOCATIONS if args.location == "all" else [args.location]
    sources   = list(SOURCES.keys()) if args.source == "all" else [args.source]

    run_pipeline(locations, sources, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
