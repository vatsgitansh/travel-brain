"""
embedder.py — Generates vector embeddings for text chunks.

Supports three providers (switchable via EMBEDDING_PROVIDER in .env):
  - "gemini"  → text-embedding-004  (768-dim, FREE via Gemini API, zero dependencies)
  - "openai"  → text-embedding-3-small  (1536-dim, paid ~$0.02/1M tokens)
  - "local"   → all-MiniLM-L6-v2   (384-dim, offline — requires PyTorch, NOT for cloud)

Cloud deployment default: EMBEDDING_PROVIDER=gemini (no torch, smallest image size)
Local dev default:        EMBEDDING_PROVIDER=local   (no API key needed)
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

# ── Lazy client singletons ────────────────────────────────────────────────────
_openai_client  = None
_local_model    = None
_gemini_client  = None


# ── Gemini Embeddings (FREE — no PyTorch, cloud-friendly) ─────────────────────

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Set it in .env or use EMBEDDING_PROVIDER=local."
            )
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_gemini_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Google Gemini text-embedding-004 (free tier)."""
    from google.genai import types as gtypes
    client = _get_gemini_client()
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
        config=gtypes.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return [emb.values for emb in result.embeddings]


# ── OpenAI Embeddings ─────────────────────────────────────────────────────────

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Either add it to .env or set "
                "EMBEDDING_PROVIDER=gemini or EMBEDDING_PROVIDER=local."
            )
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_openai_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI's text-embedding API."""
    client   = _get_openai_client()
    response = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ── Local Embeddings (NOT recommended for cloud — installs PyTorch) ───────────

def _get_local_model():
    global _local_model
    if _local_model is None:
        logger.info(f"Loading local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        logger.info("Local embedding model loaded ✓")
    return _local_model


def _embed_local_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using local sentence-transformers (requires torch)."""
    model   = _get_local_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


# ── Unified Batch Embedder ────────────────────────────────────────────────────

# Gemini allows 100 texts per batch; OpenAI allows 2048; local ~100 is safe
BATCH_SIZE = 100

def embed_texts(texts: list[str], provider: str = config.EMBEDDING_PROVIDER) -> list[list[float]]:
    """
    Embed a list of texts using the configured provider.
    Automatically batches to stay within API limits.
    """
    if not texts:
        return []

    if provider == "openai":
        embed_fn = _embed_openai_batch
    elif provider == "local":
        embed_fn = _embed_local_batch
    else:
        # Default: gemini (free, cloud-safe, no torch)
        embed_fn = _embed_gemini_batch

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.debug(f"Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} texts) via {provider}...")
        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.05)   # Small pause between batches to be API-polite

    logger.info(
        f"Embedded {len(all_embeddings)} chunks via {provider} "
        f"(dim={len(all_embeddings[0]) if all_embeddings else 0})"
    )
    return all_embeddings


# ── Embed Document Chunks ─────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for a list of tagged chunks.
    Adds `embedding` key to each chunk dict in-place.
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
