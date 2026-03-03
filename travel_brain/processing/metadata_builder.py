"""
metadata_builder.py — Auto-tags each document chunk with structured metadata
using keyword-matching rules and assigns UUIDs.

Metadata powers the future LLM's filter capabilities:
  location, budget_level, is_hidden_gem, weather_dependency,
  is_family_friendly, sentiment, tags, content_freshness
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)


# ── Region Mapping ─────────────────────────────────────────────────────────────
# Maps location keywords → specific regions within the destination
REGION_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "bali": {
        "seminyak":    ["seminyak"],
        "ubud":        ["ubud", "monkey forest", "tegalalang"],
        "uluwatu":     ["uluwatu", "ulu watu", "bingin", "padang padang"],
        "canggu":      ["canggu", "pererenan", "echo beach"],
        "nusa_penida": ["nusa penida", "nusa lembongan", "kelingking"],
        "kuta":        ["kuta"],
        "sanur":       ["sanur"],
        "amed":        ["amed", "tulamben"],
        "north_bali":  ["lovina", "singaraja", "munduk"],
        "east_bali":   ["sidemen", "candidasa", "amlapura"],
    },
    "dubai": {
        "downtown":        ["downtown", "burj khalifa", "dubai mall", "fountain"],
        "marina":          ["marina", "jbr", "jumeirah beach residence"],
        "deira":           ["deira", "gold souk", "spice souk", "old dubai"],
        "jumeirah":        ["jumeirah", "la mer"],
        "silicon_oasis":   ["silicon oasis", "academic city"],
        "al_quoz":         ["al quoz", "alserkal"],
        "hatta":           ["hatta"],
        "palm":            ["palm jumeirah", "atlantis"],
        "bur_dubai":       ["bur dubai", "al fahidi", "bastakiya"],
        "desert":          ["desert safari", "al maha", "liwa"],
    },
}


# ── Tag Taxonomy ───────────────────────────────────────────────────────────────
TAG_KEYWORDS: dict[str, list[str]] = {
    "beach":        ["beach", "shore", "coastline", "bay", "cove", "surf"],
    "food":         ["restaurant", "warung", "cafe", "street food", "local food", "eat", "cuisine", "dish"],
    "temple":       ["temple", "pura", "mosque", "shrine", "spiritual", "sacred"],
    "hiking":       ["hike", "trekking", "trek", "trail", "volcano", "mount"],
    "nightlife":    ["bar", "club", "nightlife", "party", "drinks", "cocktail"],
    "waterfall":    ["waterfall", "air terjun", "cascade"],
    "diving":       ["diving", "snorkeling", "snorkelling", "reef", "underwater"],
    "shopping":     ["shopping", "market", "souk", "bazaar", "mall"],
    "culture":      ["culture", "traditional", "local life", "heritage", "history"],
    "nature":       ["rice field", "jungle", "forest", "nature", "monkey", "wildlife"],
    "sunset":       ["sunset", "sunrise", "golden hour"],
    "instagram":    ["instagram", "insta", "photo spot", "photogenic"],
    "scam_warning": ["scam", "rip off", "avoid", "warning", "beware"],
    "budget_tip":   ["free", "cheap", "budget", "affordable", "save money"],
    "transport":    ["taxi", "grab", "gojek", "scooter", "transport", "motorbike", "uber"],
    "wifi":         ["wifi", "internet", "coworking", "cafe work", "remote work"],
}


# ── Sentiment Classifier ───────────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "amazing", "beautiful", "stunning", "incredible", "love", "perfect",
    "recommended", "worth it", "must see", "underrated", "hidden gem",
    "fantastic", "breathtaking", "peaceful", "authentic",
}
_WARNING_WORDS = {
    "scam", "avoid", "dangerous", "warning", "unsafe", "rip off",
    "beware", "overpriced", "crowded", "terrible", "worst", "closed",
    "flood", "problem", "issue", "theft",
}

def classify_sentiment(text: str) -> str:
    words  = set(text.lower().split())
    pos    = len(words & _POSITIVE_WORDS)
    neg    = len(words & _WARNING_WORDS)
    if neg > pos:
        return "warning"
    if pos > neg:
        return "positive"
    return "neutral"


# ── Individual Taggers ────────────────────────────────────────────────────────

def _tag_budget(text_lower: str) -> str:
    for level, keywords in config.BUDGET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return "unknown"


def _tag_hidden_gem(text_lower: str) -> bool:
    return any(kw in text_lower for kw in config.HIDDEN_GEM_KEYWORDS)


def _tag_weather(text_lower: str) -> str:
    for condition, keywords in config.WEATHER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return condition
    return "any"


def _tag_family_friendly(text_lower: str) -> Optional[bool]:
    if any(kw in text_lower for kw in config.FAMILY_KEYWORDS):
        return True
    if any(kw in text_lower for kw in ["adults only", "not for kids", "party", "clubbing"]):
        return False
    return None  # Unknown


def _tag_region(text_lower: str, location: str) -> Optional[str]:
    region_map = REGION_KEYWORDS.get(location, {})
    for region, keywords in region_map.items():
        if any(kw in text_lower for kw in keywords):
            return region
    return None


def _extract_tags(text_lower: str) -> list[str]:
    return [tag for tag, keywords in TAG_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)]


# ── Main Metadata Builder ──────────────────────────────────────────────────────

def build_metadata(chunk: dict) -> dict:
    """
    Tag a single chunk with the full metadata schema.
    Returns the chunk dict with a `metadata` key (Pinecone-compatible flat dict).
    """
    text       = chunk.get("chunk_text", "") or chunk.get("cleaned_text", "")
    text_lower = text.lower()
    location   = chunk.get("location", "unknown")

    # Parse content freshness from source date
    freshness = chunk.get("published_at") or chunk.get("created_utc") or chunk.get("scraped_at", "")
    if freshness:
        freshness = freshness[:10]  # ISO date YYYY-MM-DD

    metadata = {
        # ── Source provenance ─────────────────────────────────────────────────
        "source_type":    chunk.get("source_type", "unknown"),
        "source_url":     chunk.get("source_url", ""),
        "source_title":   (chunk.get("source_title", "") or "")[:200],  # Pinecone string limit

        # ── Location & Region ─────────────────────────────────────────────────
        "location":       location,
        "region":         _tag_region(text_lower, location) or "general",

        # ── Travel Intelligence Tags ──────────────────────────────────────────
        "budget_level":       _tag_budget(text_lower),
        "is_hidden_gem":      _tag_hidden_gem(text_lower),
        "is_family_friendly": _tag_family_friendly(text_lower),  # None = unknown
        "weather_dependency": _tag_weather(text_lower),
        "sentiment":          classify_sentiment(text),
        "tags":               ",".join(_extract_tags(text_lower)),   # CSV string for Pinecone

        # ── Freshness & Chunking ──────────────────────────────────────────────
        "content_freshness":  freshness,
        "chunk_index":        chunk.get("chunk_index", 0),
        "total_chunks":       chunk.get("total_chunks", 1),
        "token_count":        chunk.get("token_count", 0),
        "language":           "en",

        # ── Ingestion Timestamp ───────────────────────────────────────────────
        "ingested_at":        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    # Remove None values — Pinecone doesn't accept None
    metadata = {k: v for k, v in metadata.items() if v is not None}

    chunk["doc_id"]   = chunk.get("doc_id") or str(uuid.uuid4())
    chunk["metadata"] = metadata

    return chunk


def build_metadata_batch(chunks: list[dict]) -> list[dict]:
    """Apply metadata tagging to a batch of chunks."""
    result = [build_metadata(chunk) for chunk in chunks]

    # Stats
    hidden_gems = sum(1 for c in result if c["metadata"].get("is_hidden_gem"))
    warnings    = sum(1 for c in result if c["metadata"].get("sentiment") == "warning")
    logger.info(
        f"Metadata tagged {len(result)} chunks — "
        f"{hidden_gems} hidden gems, {warnings} travel warnings"
    )
    return result
