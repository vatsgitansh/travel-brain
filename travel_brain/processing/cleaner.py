"""
cleaner.py — Text normalization, noise removal, language detection,
and deduplication for raw scraped travel content.
"""

import hashlib
import logging
import re
import unicodedata
from typing import Optional

import ftfy
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# ── Noise Patterns ────────────────────────────────────────────────────────────

# Common patterns in scraped content that add no value.
# NOTE: No inline (?i) flags — we pass re.IGNORECASE to compile() instead.
# IMPORTANT: Use [^.\n]{0,N} (not .{0,N}) so patterns stop at sentence/line boundaries
# and don't consume legitimate travel content that follows a matched phrase.
_NOISE_PATTERNS = [
    r"\b(subscribe|like this video|hit the bell|follow us on|check out our)\b[^.\n]{0,80}",
    r"(disclaimer|affiliate link|this post (?:may )?contains? affiliate)[^.\n]{0,200}",
    r"(this (?:article|post) was (?:last )?updated|originally published)[^.\n]{0,100}",
    r"\b(cookie|privacy policy|terms of (?:service|use)|copyright)[^.\n]{0,100}",
    r"(share this|leave a comment|drop a comment|comment below)[^.\n]{0,80}",
    r"(read more|continue reading|click here to)[^.\n]{0,80}",
    r"(advertisement|sponsored post|paid partnership)[^.\n]{0,100}",
    r"https?://\S+",                          # URLs
    r"\[Music\]|\[Applause\]|\[Laughter\]",   # YouTube auto-caption artifacts
    r"\[[^\]]{0,100}\]",                      # Generic bracket noise
    r"<[^>]+>",                               # Any remaining HTML tags
    r"&[a-z]{2,6};",                          # HTML entities
]

# Compile without DOTALL so '.' doesn't eat newlines inside patterns
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE    = re.compile(r"\n{3,}")


# ── Core Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply full cleaning pipeline to raw scraped text:
    1. Fix encoding issues (ftfy)
    2. Normalize unicode (NFC)
    3. Remove noise patterns
    4. Normalize whitespace
    """
    if not text or not text.strip():
        return ""

    # 1. Fix mojibake, smart quotes, broken unicode
    text = ftfy.fix_text(text)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 3. Remove noise
    text = _NOISE_RE.sub(" ", text)

    # 4. Whitespace normalization
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── Language Detection ────────────────────────────────────────────────────────

def is_english(text: str, sample_chars: int = 500) -> bool:
    """
    Returns True if the text is detected as English.
    Samples first N chars for speed.
    """
    sample = text[:sample_chars].strip()
    if len(sample) < 50:
        return True   # Too short to reliably detect, assume OK
    try:
        lang = detect(sample)
        return lang == "en"
    except LangDetectException:
        return True   # Assume English on detection failure


# ── Deduplication ─────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """
    Generate a SHA256 hash of normalized text for deduplication.
    Collapses all whitespace before hashing so minor formatting differences
    don't create false uniqueness.
    """
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DeduplicationTracker:
    """
    Tracks seen content hashes across a scraping session to deduplicate
    documents before they enter the processing pipeline.
    """

    def __init__(self):
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = content_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def reset(self):
        self._seen.clear()

    @property
    def count(self) -> int:
        return len(self._seen)


# ── Document Cleaner ───────────────────────────────────────────────────────────

def clean_document(doc: dict, dedup: Optional[DeduplicationTracker] = None) -> Optional[dict]:
    """
    Clean a single raw document dictionary in-place.
    Returns None if the document should be discarded (too short, not English, duplicate).
    """
    raw_text = doc.get("raw_text", "")

    # Step 1: Clean
    cleaned = clean_text(raw_text)
    if len(cleaned) < 50:
        logger.debug(f"Discarded (too short after cleaning): {doc.get('source_url', '?')}")
        return None

    # Step 2: Language filter
    if not is_english(cleaned):
        logger.debug(f"Discarded (non-English): {doc.get('source_url', '?')}")
        return None

    # Step 3: Deduplication
    if dedup and dedup.is_duplicate(cleaned):
        logger.debug(f"Discarded (duplicate): {doc.get('source_url', '?')}")
        return None

    doc["cleaned_text"] = cleaned
    doc["text_hash"]    = content_hash(cleaned)
    doc["char_count"]   = len(cleaned)

    return doc


def clean_documents(documents: list[dict]) -> list[dict]:
    """
    Clean a batch of raw documents.
    Performs session-level deduplication across the entire batch.
    Returns only clean, unique, English documents.
    """
    dedup   = DeduplicationTracker()
    cleaned = []
    stats   = {"total": len(documents), "discarded_short": 0, "discarded_lang": 0, "discarded_dup": 0}

    for doc in documents:
        result = clean_document(doc, dedup)
        if result:
            cleaned.append(result)
        else:
            pass  # Counting is implicit via dedup tracker

    kept       = len(cleaned)
    discarded  = len(documents) - kept
    logger.info(f"Cleaned {kept}/{len(documents)} documents ({discarded} discarded)")
    return cleaned
