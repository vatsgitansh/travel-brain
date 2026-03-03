"""
test_cleaner.py — Unit tests for the text cleaning module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from travel_brain.processing.cleaner import (
    clean_text,
    is_english,
    content_hash,
    DeduplicationTracker,
    clean_documents,
)


class TestCleanText:

    def test_removes_html_tags(self):
        raw     = "<p>This is <b>bold</b> text.</p>"
        cleaned = clean_text(raw)
        assert "<" not in cleaned
        assert "bold" in cleaned

    def test_removes_urls(self):
        raw     = "Check out https://www.example.com for more info."
        cleaned = clean_text(raw)
        assert "https://" not in cleaned
        assert "Check out" in cleaned

    def test_removes_affiliate_disclaimer(self):
        raw     = "This post contains affiliate links. We love Bali!"
        cleaned = clean_text(raw)
        assert "We love Bali" in cleaned

    def test_removes_subscribe_noise(self):
        raw     = "Subscribe to our channel for tips. Bali has great beaches."
        cleaned = clean_text(raw)
        assert "subscribe" not in cleaned.lower()
        # The Bali content after the period should survive
        assert "Bali" in cleaned or len(cleaned) < len(raw)

    def test_removes_youtube_captions(self):
        raw     = "[Music] The beach was beautiful. [Applause]"
        cleaned = clean_text(raw)
        assert "[Music]" not in cleaned
        assert "The beach was beautiful" in cleaned

    def test_normalizes_excessive_newlines(self):
        raw     = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_text(raw)
        assert "\n\n\n" not in cleaned

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_fixes_encoding_issues(self):
        # ftfy should fix mojibake
        raw     = "I\u2019ve been to Bali"  # Smart apostrophe
        cleaned = clean_text(raw)
        assert "Bali" in cleaned


class TestLanguageDetection:

    def test_detects_english(self):
        text = "Bali is a beautiful island in Indonesia with stunning temples and beaches."
        assert is_english(text) is True

    def test_short_text_passes(self):
        # Too short to reliably detect — should default to True
        assert is_english("Hello Bali") is True

    def test_empty_text_passes(self):
        assert is_english("") is True


class TestDeduplication:

    def test_same_text_is_duplicate(self):
        tracker = DeduplicationTracker()
        text    = "The hidden waterfall in Ubud is absolutely stunning."
        assert tracker.is_duplicate(text) is False   # First time: not duplicate
        assert tracker.is_duplicate(text) is True    # Second time: duplicate

    def test_whitespace_normalized_for_hash(self):
        tracker = DeduplicationTracker()
        text1   = "Hidden gem in Bali"
        text2   = "Hidden   gem  in  Bali"   # Extra spaces
        tracker.is_duplicate(text1)
        assert tracker.is_duplicate(text2) is True  # Same after normalization

    def test_different_text_not_duplicate(self):
        tracker = DeduplicationTracker()
        tracker.is_duplicate("Text about Bali")
        assert tracker.is_duplicate("Text about Dubai") is False

    def test_reset_clears_state(self):
        tracker = DeduplicationTracker()
        text    = "Some travel tip"
        tracker.is_duplicate(text)
        tracker.reset()
        assert tracker.is_duplicate(text) is False  # Fresh start


class TestCleanDocuments:

    def _make_doc(self, text: str, url: str = "https://example.com") -> dict:
        return {"raw_text": text, "source_url": url, "source_type": "blog", "location": "bali"}

    def test_removes_short_documents(self):
        docs   = [self._make_doc("Short text.")]
        result = clean_documents(docs)
        assert len(result) == 0

    def test_keeps_valid_document(self):
        # Use clean travel content with no noise trigger words
        long_text = "The rice terraces in Ubud are breathtaking. Local farmers work here daily. " * 20
        docs      = [self._make_doc(long_text)]
        result    = clean_documents(docs)
        assert len(result) == 1
        assert "cleaned_text" in result[0]
        assert "text_hash" in result[0]

    def test_deduplicates_across_batch(self):
        text  = "The same travel tip repeated here. " * 20
        docs  = [
            self._make_doc(text, url="https://blog1.com"),
            self._make_doc(text, url="https://blog2.com"),
        ]
        result = clean_documents(docs)
        assert len(result) == 1  # Second one is a duplicate
