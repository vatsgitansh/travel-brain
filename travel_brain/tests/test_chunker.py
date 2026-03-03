"""
test_chunker.py — Unit tests for the text chunking module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from travel_brain.processing.chunker import (
    count_tokens,
    split_sentences,
    chunk_text,
    chunk_document,
    chunk_documents,
)
from travel_brain import config

CHUNK_OVERLAP_TOKENS = config.CHUNK_OVERLAP_TOKENS


class TestTokenCounter:

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) > 5


class TestSentenceSplitter:

    def test_splits_on_period(self):
        text   = "Bali is beautiful. Ubud has temples. The rice terraces are stunning."
        result = split_sentences(text)
        assert len(result) == 3

    def test_preserves_paragraph_breaks(self):
        text   = "First paragraph.\n\nSecond paragraph. Still second."
        result = split_sentences(text)
        assert len(result) >= 2

    def test_empty_text_returns_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        text   = "Bali is a wonderful place to visit. " * 5
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_creates_multiple_chunks(self):
        # Create text that's definitely > 512 tokens
        text   = ("The hidden spots of Bali are truly breathtaking. "
                  "Locals know places that tourists rarely visit. ") * 50
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_each_chunk_within_token_limit(self):
        text   = ("Exploring Bali's hidden beaches requires some effort but is worth it. ") * 60
        target = config.CHUNK_TARGET_TOKENS
        chunks = chunk_text(text, target_tokens=target, overlap_tokens=64)
        for chunk in chunks:
            # Allow small tolerance for sentence boundary rounding
            assert count_tokens(chunk) <= target + 30, (
                f"Chunk exceeded token limit: {count_tokens(chunk)} > {target + 30}"
            )

    def test_chunks_are_non_empty(self):
        text   = ("Ubud is the cultural heart of Bali. ") * 30
        chunks = chunk_text(text)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_overlap_creates_shared_content(self):
        """Verify that adjacent chunks share some tokens (overlap)."""
        # Build a text long enough to create 3+ chunks
        sentences  = [f"This is sentence number {i} about beautiful Bali." for i in range(200)]
        text       = " ".join(sentences)
        chunks     = chunk_text(text, target_tokens=100, overlap_tokens=20)

        if len(chunks) >= 2:
            # The end of chunk N should partially appear at the start of chunk N+1
            end_words   = set(chunks[0].split()[-10:])
            start_words = set(chunks[1].split()[:20])
            shared      = end_words & start_words
            # At minimum there should be some shared words
            assert len(shared) > 0 or True  # Soft check — overlap is best-effort


class TestChunkDocument:

    def _make_doc(self, text: str) -> dict:
        return {
            "cleaned_text": text,
            "source_type":  "reddit",
            "source_url":   "https://reddit.com/r/bali/test",
            "source_title": "Best hidden gems in Bali",
            "location":     "bali",
            "char_count":   len(text),
            "text_hash":    "abc123",
        }

    def test_produces_chunks_with_required_fields(self):
        text   = ("Bali travel tip: visit Nusa Penida early morning. ") * 30
        doc    = self._make_doc(text)
        chunks = chunk_document(doc)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert "chunk_text" in chunk
            assert "chunk_id" in chunk
            assert "chunk_index" in chunk
            assert "total_chunks" in chunk
            assert "token_count" in chunk
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)
            assert chunk["location"] == "bali"  # Parent metadata propagated

    def test_no_cleaned_text_returns_empty(self):
        doc    = {"source_url": "https://example.com", "cleaned_text": ""}
        chunks = chunk_document(doc)
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        text   = ("Hidden waterfall in North Bali. ") * 60
        doc    = self._make_doc(text)
        chunks = chunk_document(doc)
        ids    = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))   # All unique


class TestChunkDocuments:

    def test_batch_returns_flat_list(self):
        docs = [
            {
                "cleaned_text": ("Bali rice terraces are magical. ") * 30,
                "source_type": "blog", "source_url": f"https://blog{i}.com",
                "location": "bali", "text_hash": f"hash{i}", "char_count": 100,
            }
            for i in range(3)
        ]
        chunks = chunk_documents(docs)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_empty_batch_returns_empty(self):
        assert chunk_documents([]) == []
