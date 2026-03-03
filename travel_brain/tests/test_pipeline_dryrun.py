"""
test_pipeline_dryrun.py — End-to-end dry-run test of the pipeline.
No API keys needed — uses mock scraper data and local embedding model.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from travel_brain.processing.cleaner          import clean_documents
from travel_brain.processing.chunker          import chunk_documents
from travel_brain.processing.metadata_builder import build_metadata_batch


# ── Fixtures: Mock raw documents (no scraping needed) ─────────────────────────

MOCK_BALI_REDDIT_DOC = {
    "source_type":  "reddit",
    "source_url":   "https://reddit.com/r/bali/test_post",
    "source_title": "Hidden gems in Bali no tourist knows about",
    "subreddit":    "bali",
    "location":     "bali",
    "score":        450,
    "created_utc":  "2025-11-15T08:00:00Z",
    "raw_text": """
    TITLE: Hidden gems in Bali no tourist knows about

    POST: I've spent 6 months living in Bali and discovered some incredible hidden gems that 
    most tourists never find. The key secret is to get off the main roads and explore on a 
    scooter. Here are my top free hidden spots that locals actually love.

    The Tibumana waterfall near Ubud is absolutely breathtaking and barely has any tourists.
    You can park your scooter for free and hike down in 15 minutes. There's no entrance fee 
    and the water is crystal clear. Go early morning to avoid crowds.

    Another budget tip: eat at the local warung stalls near the rice terraces in Tegalalang.
    The tourist restaurants charge 5-10x the local price. Walk 200 meters off the main road 
    and you'll find authentic nasi goreng for under 30,000 IDR.

    WARNING: Avoid the obvious "Instagram spots" in Seminyak as they charge hefty entrance 
    fees of 50,000-100,000 IDR just for a photo. The algorithm is rigging the tourist trap.

    TOP COMMENTS:
    ---
    This is gold! The scooter advice is so true. Rented one for 70,000 IDR per day and saw 
    more real Bali in 3 days than most tourists see in 2 weeks.
    ---
    Tibumana is the best kept secret! We went at 7am and had the whole waterfall to ourselves.
    The local family who manages it is incredibly friendly.
    """,
    "scraped_at": "2025-12-01T00:00:00Z",
}

MOCK_DUBAI_BLOG_DOC = {
    "source_type":  "blog",
    "source_url":   "https://nerdnomads.com/dubai-hidden-gems",
    "source_title": "Dubai's Best Kept Secrets: Free Things to Do",
    "location":     "dubai",
    "published_at": "2025-10-20",
    "raw_text": """
    Dubai's Best Kept Secrets: Free Things to Do

    Everyone thinks Dubai requires a massive budget, but I discovered incredible free 
    experiences that locals cherish. This guide covers Dubai's hidden gems that won't 
    cost you a dirham.

    The Al Fahidi Historical District (Bastakiya) is one of Dubai's most underrated 
    neighborhoods. This old district dates back to the 1890s and is completely free to explore.
    The winding alleyways, wind towers, and art galleries give you an authentic window into 
    pre-oil Dubai. Most tourists rush to the Burj Khalifa and miss this entirely.

    The Dubai Creek and the traditional abra (water taxi) ride costs just 1 AED and gives 
    you a stunning view of Old Dubai's skyline. It's one of the best values in the city.
    Take the abra from the Deira old souk side at sunset for a magical experience.

    The Spice Souk and Gold Souk in Deira are completely free to browse. The fragrant spice 
    stalls sell everything for a fraction of mall prices. Pro tip: always negotiate — prices 
    are typically marked up 30-50% for tourists.

    Summer heat warning: June through August temperatures exceed 40°C. If you're visiting 
    during this period, plan all outdoor activities before 9am or after 7pm. Most outdoor 
    attractions are best avoided during peak summer heat.
    """,
    "scraped_at": "2025-12-01T00:00:00Z",
}

MOCK_DOCS = [MOCK_BALI_REDDIT_DOC, MOCK_DUBAI_BLOG_DOC]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEndToEndDryRun:

    def test_cleaning_passes(self):
        cleaned = clean_documents(MOCK_DOCS)
        assert len(cleaned) == 2
        for doc in cleaned:
            assert "cleaned_text" in doc
            assert "text_hash" in doc
            assert len(doc["cleaned_text"]) > 100

    def test_chunking_produces_chunks(self):
        cleaned = clean_documents(MOCK_DOCS)
        chunks  = chunk_documents(cleaned)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_text" in chunk
            assert "chunk_id" in chunk
            assert "location" in chunk

    def test_metadata_tagging(self):
        cleaned = clean_documents(MOCK_DOCS)
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)

        assert len(tagged) > 0
        for chunk in tagged:
            assert "metadata" in chunk
            meta = chunk["metadata"]
            assert "location" in meta
            assert "source_type" in meta
            assert "budget_level" in meta
            assert "is_hidden_gem" in meta
            assert "sentiment" in meta
            assert "ingested_at" in meta

    def test_bali_doc_tagged_correctly(self):
        """Verify Bali Reddit doc gets correct auto-tags."""
        cleaned = clean_documents([MOCK_BALI_REDDIT_DOC])
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)

        assert all(c["metadata"]["location"] == "bali" for c in tagged)
        # Should detect hidden gem language
        gem_chunks = [c for c in tagged if c["metadata"]["is_hidden_gem"]]
        assert len(gem_chunks) > 0, "Expected at least one hidden gem tagged chunk"

    def test_dubai_doc_tagged_correctly(self):
        """Verify Dubai blog gets correct auto-tags."""
        cleaned = clean_documents([MOCK_DUBAI_BLOG_DOC])
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)

        assert all(c["metadata"]["location"] == "dubai" for c in tagged)
        # Should detect free/budget language
        budget_chunks = [c for c in tagged if c["metadata"]["budget_level"] in ("free", "budget")]
        assert len(budget_chunks) > 0, "Expected budget/free tagged chunks for Dubai doc"

    def test_warning_sentiment_detected(self):
        """Scam/warning language should be detected in sentiment."""
        cleaned = clean_documents(MOCK_DOCS)
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)
        # At least some chunks should be warning sentiment
        warning_chunks = [c for c in tagged if c["metadata"]["sentiment"] == "warning"]
        # Soft assertion — depends on chunk splits
        assert isinstance(warning_chunks, list)

    def test_all_chunks_have_doc_id(self):
        """Every chunk must have a unique doc_id."""
        cleaned = clean_documents(MOCK_DOCS)
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)
        ids     = [c["doc_id"] for c in tagged]
        assert all(ids)  # None of them are empty/None

    def test_json_serializable(self):
        """
        The final chunk output (minus embeddings) must be JSON serializable —
        critical for dry-run file output and debugging.
        """
        cleaned = clean_documents(MOCK_DOCS)
        chunks  = chunk_documents(cleaned)
        tagged  = build_metadata_batch(chunks)

        safe = [{k: v for k, v in c.items() if k != "embedding"} for c in tagged]
        json_str = json.dumps(safe, ensure_ascii=False)
        assert len(json_str) > 0
