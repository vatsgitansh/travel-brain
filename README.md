# 🧠 Travel Brain — Data Ingestion Pipeline

> **The proprietary data engine powering our Travel LLM.**
> Automatically scrapes, cleans, chunks, embeds, and stores travel intelligence for Dubai and Bali into a vector database — ready for RAG retrieval.

---

## Architecture Overview

```
YouTube Transcripts ──┐
Reddit Posts       ──┤──▶ Clean ──▶ Chunk ──▶ Embed ──▶ Vector DB (RAG)
Niche Travel Blogs ──┘         ↕ Tag Metadata
```

### Tech Stack

| Layer         | Technology                                    |
|---------------|-----------------------------------------------|
| Scraping      | PRAW, youtube-transcript-api, BeautifulSoup4  |
| Text Cleaning | ftfy, langdetect, regex                       |
| Chunking      | tiktoken (cl100k_base, sentence-aware)        |
| Embeddings    | OpenAI `text-embedding-3-small` or local `all-MiniLM-L6-v2` |
| Vector DB     | **Pinecone** (prod) / **ChromaDB** (local/offline) |
| Scheduling    | APScheduler (weekly cron)                     |

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/poo09/Desktop/Feed_maker
python -m venv venv
source venv/bin/activate
pip install -r travel_brain/requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

**Minimum required for local testing (no API keys needed):**
```env
EMBEDDING_PROVIDER=local   # Uses free sentence-transformers model
VECTOR_DB_PROVIDER=chroma  # Uses local ChromaDB — no cloud needed
SCRAPE_LIMIT=10
```

### 3. Run Tests (no API keys needed)

```bash
python -m pytest travel_brain/tests/ -v
```

### 4. Run the Pipeline

```bash
# Dry run — scrape Reddit for Bali, save chunks locally (no DB write)
python travel_brain/pipeline.py --location bali --source reddit --dry-run

# Full run — Reddit for Dubai, writes to local ChromaDB
python travel_brain/pipeline.py --location dubai --source reddit

# All sources, all locations
python travel_brain/pipeline.py
```

### 5. Start the Scheduler (Weekly Auto-refresh)

```bash
python travel_brain/scheduler.py
```

---

## Project Structure

```
travel_brain/
├── config.py                   # All settings & scraping targets
├── pipeline.py                 # Master orchestrator CLI
├── scheduler.py                # Weekly cron jobs
│
├── scrapers/
│   ├── youtube_scraper.py      # YouTube Data API + transcript extraction
│   ├── reddit_scraper.py       # PRAW — r/bali, r/travel, r/shoestring etc.
│   └── blog_scraper.py         # BeautifulSoup blog crawler
│
├── processing/
│   ├── cleaner.py              # Noise removal, dedup, language filter
│   ├── chunker.py              # 512-token semantic chunking with overlap
│   ├── embedder.py             # OpenAI or local sentence-transformers
│   └── metadata_builder.py    # Auto-tags all chunks with rich metadata
│
├── vectordb/
│   ├── base_client.py          # Abstract interface (swap DBs with zero code change)
│   ├── pinecone_client.py      # Production — Pinecone Serverless
│   └── chroma_client.py        # Local/offline — ChromaDB
│
└── tests/
    ├── test_cleaner.py
    ├── test_chunker.py
    └── test_pipeline_dryrun.py
```

---

## Metadata Schema

Every chunk stored in the vector DB carries this metadata:

```json
{
  "source_type": "reddit | youtube | blog",
  "source_url": "https://...",
  "source_title": "Hidden gems in Bali no tourist knows about",
  "location": "bali | dubai",
  "region": "uluwatu | ubud | deira | marina | ...",
  "budget_level": "free | budget | mid | luxury",
  "is_hidden_gem": true,
  "is_family_friendly": true,
  "weather_dependency": "dry_season | avoid_monsoon | summer_heat | any",
  "sentiment": "positive | warning | neutral",
  "tags": "beach,sunset,no_crowds,local_food",
  "content_freshness": "2025-11-15",
  "chunk_index": 0,
  "total_chunks": 3,
  "token_count": 487,
  "language": "en",
  "ingested_at": "2026-02-28"
}
```

### Querying with Filters (Pinecone example)

```python
from travel_brain.vectordb.pinecone_client import PineconeClient
from travel_brain.processing.embedder import embed_texts

db    = PineconeClient()
query = "free things to do in Dubai that locals love"
vec   = embed_texts([query])[0]

results = db.query(
    query_embedding=vec,
    top_k=5,
    namespace="dubai",
    filter={
        "budget_level": {"$in": ["free", "budget"]},
        "is_hidden_gem": {"$eq": True},
        "sentiment": {"$ne": "warning"},
    }
)
```

---

## API Keys Guide

| Key | Where to get | Cost |
|-----|-------------|------|
| `OPENAI_API_KEY` | platform.openai.com | ~$0.02/1M tokens (skip with `EMBEDDING_PROVIDER=local`) |
| `PINECONE_API_KEY` | app.pinecone.io | Free tier: 1 index, 100K vectors |
| `REDDIT_CLIENT_ID/SECRET` | reddit.com/prefs/apps | Free — create a "script" app |
| `YOUTUBE_API_KEY` | console.cloud.google.com | Free tier: 10,000 units/day |

> **Development tip:** Set `EMBEDDING_PROVIDER=local` and `VECTOR_DB_PROVIDER=chroma` to run the entire pipeline for free without any API keys.

---

## Roadmap

- [x] Phase 1: Data ingestion pipeline (this repo)
- [ ] Phase 2: Query API (FastAPI wrapper over vector DB)
- [ ] Phase 3: Fine-tuning dataset export (JSONL format for LLM training)
- [ ] Phase 4: Quantized Travel LLM (GGUF / ONNX for mobile deployment)
- [ ] Phase 5: Offline mobile app with embedded ChromaDB + local LLM
