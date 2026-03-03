"""
chunker.py — Semantic text chunking with token-aware splitting and overlap.
Splits cleaned travel documents into embedding-ready chunks of ~512 tokens
with a 64-token overlap to preserve context at chunk boundaries.
"""

import logging
import re
from typing import Iterator

import tiktoken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)

# ── Tokenizer (cl100k_base = GPT-4 / text-embedding-3 compatible) ─────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Token Utilities ───────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def token_slice(text: str, start: int, end: int) -> str:
    """Return the text slice corresponding to tokens[start:end]."""
    tokens = _TOKENIZER.encode(text)
    return _TOKENIZER.decode(tokens[start:end])


# ── Sentence Splitter ─────────────────────────────────────────────────────────

_SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Preserves paragraph breaks as sentence boundaries."""
    # First split on paragraph boundaries
    paragraphs = text.split("\n\n")
    sentences  = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Then split each paragraph into sentences
        parts = _SENTENCE_ENDINGS.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


# ── Core Chunker ──────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split `text` into chunks of approximately `target_tokens` tokens each,
    with `overlap_tokens` token overlap between consecutive chunks.

    Algorithm:
    1. Split text into sentences
    2. Greedily accumulate sentences until the chunk would exceed target_tokens
    3. Emit the chunk, then backtrack `overlap_tokens` worth of sentences
       to create the overlap for the next chunk
    """
    if not text.strip():
        return []

    # Fast path: entire text fits in one chunk
    if count_tokens(text) <= target_tokens:
        return [text.strip()]

    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens: int = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # Edge case: single sentence longer than target → emit as standalone chunk
        if sent_tokens > target_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens    = 0
            # Split the monster sentence by raw token windows
            for sub_chunk in _token_window_split(sentence, target_tokens, overlap_tokens):
                chunks.append(sub_chunk)
            continue

        # If adding this sentence would exceed target, emit and start new chunk
        if current_tokens + sent_tokens > target_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Build overlap: backtrack sentences until we have ~overlap_tokens
            overlap_sents: list[str] = []
            overlap_count: int       = 0
            for s in reversed(current_sentences):
                s_tok = count_tokens(s)
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_tok

            current_sentences = overlap_sents
            current_tokens    = overlap_count

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Emit final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c.strip() for c in chunks if c.strip()]


def _token_window_split(text: str, target: int, overlap: int) -> Iterator[str]:
    """Fallback: split a very long sentence using raw token windows."""
    tokens = _TOKENIZER.encode(text)
    step   = target - overlap
    for i in range(0, len(tokens), step):
        window = tokens[i : i + target]
        if window:
            yield _TOKENIZER.decode(window)


# ── Document Chunker ──────────────────────────────────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """
    Split a cleaned document into chunks.
    Each chunk inherits all metadata from the parent document,
    plus gets its own `chunk_index` and `total_chunks` fields.
    """
    text = doc.get("cleaned_text", "")
    if not text:
        logger.warning(f"No cleaned_text in document: {doc.get('source_url', '?')}")
        return []

    raw_chunks = chunk_text(text)
    if not raw_chunks:
        return []

    total = len(raw_chunks)
    chunks: list[dict] = []

    for i, chunk_text_content in enumerate(raw_chunks):
        chunk = {
            # Propagate all parent metadata
            **{k: v for k, v in doc.items() if k not in ("raw_text", "cleaned_text")},
            # ── Chunk-specific fields ─────────────────────────────────────
            "chunk_id":     f"{doc.get('text_hash', 'unknown')}_chunk_{i}",
            "chunk_index":  i,
            "total_chunks": total,
            "chunk_text":   chunk_text_content,
            "token_count":  count_tokens(chunk_text_content),
        }
        chunks.append(chunk)

    logger.debug(
        f"Chunked '{doc.get('source_title', '?')}' into {total} chunks "
        f"({doc.get('char_count', 0)} chars)"
    )
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk a batch of cleaned documents. Returns a flat list of all chunks."""
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    token_counts = [c["token_count"] for c in all_chunks]
    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks from {len(documents)} documents "
            f"(avg {avg:.0f} tokens/chunk, max {max(token_counts)})"
        )
    return all_chunks
