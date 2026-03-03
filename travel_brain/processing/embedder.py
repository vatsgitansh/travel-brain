"""
embedder.py — Generates vector embeddings for text chunks.

Supports two providers (switchable via .env):
  - "openai"  → text-embedding-3-small  (1536-dim, API cost ~$0.02/1M tokens)
  - "local"   → all-MiniLM-L6-v2       (384-dim, fully offline, zero cost)

The local model is the default for dev and critical for our offline phone roadmap.
"""

import logging
import time
from typing import Union

from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)

# ── Lazy Imports (only load what's needed) ────────────────────────────────────

_openai_client   = None
_local_model     = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Either add it to .env or set "
                "EMBEDDING_PROVIDER=local to use the free local model."
            )
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        logger.info(f"Loading local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        logger.info("Local embedding model loaded ✓")
    return _local_model


# ── OpenAI Embeddings ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_openai_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI's API."""
    client   = _get_openai_client()
    response = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    # Response order matches input order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ── Local Embeddings ──────────────────────────────────────────────────────────

def _embed_local_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using local sentence-transformers."""
    model  = _get_local_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


# ── Unified Batch Embedder ────────────────────────────────────────────────────

BATCH_SIZE = 100   # OpenAI allows up to 2048 inputs per request; local ~100 is safe

def embed_texts(texts: list[str], provider: str = config.EMBEDDING_PROVIDER) -> list[list[float]]:
    """
    Embed a list of texts using the configured provider.
    Automatically batches to stay within API limits.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    embed_fn = _embed_openai_batch if provider == "openai" else _embed_local_batch

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.debug(f"Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} texts) via {provider}...")

        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)

        if provider == "openai" and i + BATCH_SIZE < len(texts):
            time.sleep(0.1)  # Gentle rate limiting

    logger.info(f"Embedded {len(all_embeddings)} chunks via {provider} (dim={len(all_embeddings[0]) if all_embeddings else 0})")
    return all_embeddings


# ── Embed Document Chunks ─────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for a list of tagged chunks.
    Adds `embedding` key to each chunk dict.
    Returns chunks with embeddings attached.
    """
    texts = [chunk["chunk_text"] for chunk in chunks]

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks
