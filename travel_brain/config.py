"""
config.py — Centralized configuration for the Travel Brain pipeline.
All settings are loaded from the .env file (or environment variables).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = _ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
CHROMA_DIR      = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

for _d in [RAW_DIR, PROCESSED_DIR, CHROMA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT  = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "travel-brain")
REDDIT_CLIENT_ID      = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET  = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT     = os.getenv("REDDIT_USER_AGENT", "TravelBrainBot/1.0")
YOUTUBE_API_KEY       = os.getenv("YOUTUBE_API_KEY", "")

# ── Provider Selection ────────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")   # "openai" | "local"
VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "chroma")  # "pinecone" | "chroma"

# LLM for chat — auto-detected from available keys, or set explicitly
# Options: "gemini" | "openai" | "" (no LLM, fallback to retrieval-only)
_llm_explicit = os.getenv("LLM_PROVIDER", "").lower()
if _llm_explicit:
    LLM_PROVIDER = _llm_explicit
elif GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AI..."):
    LLM_PROVIDER = "gemini"
elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-your"):
    LLM_PROVIDER = "openai"
else:
    LLM_PROVIDER = ""

# ── Embedding Models ──────────────────────────────────────────────────────────
# Provider priority:
#   gemini → free, cloud-safe, no PyTorch (RECOMMENDED for Railway)
#   openai → paid, high quality
#   local  → offline only, requires torch (~3GB) — DO NOT use on cloud
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")  # default: gemini

OPENAI_EMBEDDING_MODEL  = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM    = 1536

GEMINI_EMBEDDING_MODEL  = "models/text-embedding-004"
GEMINI_EMBEDDING_DIM    = 768

LOCAL_EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIM     = 384

EMBEDDING_DIM = (
    OPENAI_EMBEDDING_DIM if EMBEDDING_PROVIDER == "openai"
    else GEMINI_EMBEDDING_DIM if EMBEDDING_PROVIDER == "gemini"
    else LOCAL_EMBEDDING_DIM
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# ── Scraper Targets ───────────────────────────────────────────────────────────
SCRAPE_LIMIT = int(os.getenv("SCRAPE_LIMIT", "500"))

LOCATIONS = ["bali", "dubai"]

# YouTube search queries per location — 20 per location for coverage
YOUTUBE_QUERIES: dict[str, list[str]] = {
    "bali": [
        "hidden gems Bali 2025",
        "Bali off the beaten path",
        "Bali travel tips locals only",
        "Bali on a budget backpacker",
        "Bali hidden beaches 2025",
        "Bali vlog expat life",
        "Bali without tourists",
        "Bali digital nomad guide",
        "Bali food guide local restaurants",
        "Bali scams avoid tourists",
        "Bali transport scooter guide",
        "Bali best time to visit rainy season",
        "Bali Ubud travel guide",
        "Bali Canggu guide 2025",
        "Bali Seminyak Kuta guide",
        "Bali temples culture etiquette",
        "Bali nightlife party guide",
        "Bali meditation retreat wellness",
        "Bali surfing beginner guide",
        "moving to Bali expat guide 2025",
    ],
    "dubai": [
        "hidden gems Dubai 2025",
        "Dubai without spending money",
        "Dubai free things to do",
        "Dubai off the beaten path",
        "Dubai budget travel guide",
        "Dubai travel tips locals",
        "Dubai secrets vlog",
        "Dubai cheap eats guide",
        "Dubai desert safari experience",
        "Dubai old town Deira guide",
        "Dubai nightlife bars clubs",
        "Dubai scams avoid tourists",
        "Dubai transport metro guide",
        "Dubai best beaches free",
        "Dubai luxury experience affordable",
        "Dubai food street eats guide",
        "Dubai culture etiquette rules",
        "Dubai shopping mall guide",
        "living in Dubai expat guide 2025",
        "Dubai vs Abu Dhabi travel comparison",
    ],
}

# Reddit subreddits to scrape per location
REDDIT_SUBREDDITS: dict[str, list[str]] = {
    "bali": ["bali", "travel", "solotravel", "shoestring", "digitalnomad", "expats"],
    "dubai": ["dubai", "travel", "solotravel", "shoestring", "UAE", "digitalnomad"],
}

# Reddit search keywords per location
REDDIT_SEARCH_TERMS: dict[str, list[str]] = {
    "bali": ["hidden gem", "off beaten", "local tip", "avoid tourist", "budget bali", "secret spot"],
    "dubai": ["hidden gem", "free dubai", "budget dubai", "local tip", "avoid tourist", "cheap dubai"],
}

# Curated niche travel blogs — 60+ seed URLs per location for deep crawling
BLOG_SEEDS: dict[str, list[str]] = {
    "bali": [
        # ── General Bali travel guides ─────────────────────────────────────────
        "https://www.nomadicmatt.com/travel-guides/indonesia-travel-tips/bali/",
        "https://www.lonelyplanet.com/indonesia/bali",
        "https://www.roughguides.com/indonesia/bali/",
        "https://www.cntraveler.com/destinations/bali",
        "https://www.timeout.com/bali",
        # ── Bali-specific blogs ────────────────────────────────────────────────
        "https://balibibles.com/",
        "https://balistyle.net/",
        "https://www.bali-indonesia.com/bali/",
        "https://baliguidemap.com/",
        "https://bali.com/blog/",
        "https://www.baliexpat.com/",
        "https://www.balitravels.com/blog/",
        "https://www.balivillas.com/article",
        "https://discoverbali.id/blog/",
        # ── Indie / backpacker blogs ───────────────────────────────────────────
        "https://www.indietraveller.co/bali/",
        "https://www.lostwithpurpose.com/?s=bali",
        "https://www.nerdnomads.com/?s=bali",
        "https://www.theplanetd.com/?s=bali",
        "https://www.adventurouskate.com/?s=bali",
        "https://www.journeyera.com/?s=bali",
        "https://thebrokebackpacker.com/bali-travel-guide/",
        "https://www.backpackertravel.com.au/?s=bali",
        "https://www.breathedreamgo.com/?s=bali",
        "https://www.travellingbuzz.com/?s=bali",
        "https://www.alongdustyroads.com/?s=bali",
        "https://www.travelwithbender.com/?s=bali",
        "https://hellosomewhere.com/?s=bali",
        # ── Food & budget ──────────────────────────────────────────────────────
        "https://www.thehoneycombers.com/bali/eat-drink/",
        "https://www.spotlightbali.com/dining/",
        "https://balifoodguide.com/",
        # ── Digital nomad ─────────────────────────────────────────────────────
        "https://nomadlist.com/bali",
        "https://www.goatsontheroad.com/?s=bali",
        "https://expertvagabond.com/?s=bali",
        "https://www.danflyingsolo.com/?s=bali",
        "https://youngadventuress.com/?s=bali",
        # ── Ubud & Canggu specific ─────────────────────────────────────────────
        "https://www.ubudwritersfestival.com/",
        "https://casavacationbali.com/blog/",
        "https://canggu.guide/",
        # ── Wellness & yoga ────────────────────────────────────────────────────
        "https://www.yogabali.com/blog/",
        "https://www.balispirit.com/blog/",
        # ── Video / pop culture blogs ──────────────────────────────────────────
        "https://www.heysamyoussef.com/?s=bali",
        "https://www.sorellaboutique.com/?s=bali",
        # ── Magazine and listicles ─────────────────────────────────────────────
        "https://www.tripsavvy.com/bali-4159967",
        "https://www.tripzilla.com/tag/bali",
        "https://thesavvybackpacker.com/?s=bali",
        "https://www.worldpackers.com/articles/bali",
        "https://theplanetd.com/bali-travel-guide/",
        "https://www.nomadicchica.com/?s=bali",
        "https://www.migrationology.com/?s=bali",
        "https://www.eatyourworld.com/?s=bali",
        "https://www.foodtourist.com/?s=bali",
        "https://www.helenabunntravel.com/?s=bali",
        "https://www.bruised-passports.com/?s=bali",
        "https://theplanetleft.com/?s=bali",
        "https://www.handluggageonly.co.uk/?s=bali",
        "https://www.mylifesamovie.com/?s=bali",
    ],
    "dubai": [
        # ── General Dubai travel guides ────────────────────────────────────────
        "https://www.nomadicmatt.com/travel-guides/united-arab-emirates-travel-tips/dubai/",
        "https://www.lonelyplanet.com/united-arab-emirates/dubai",
        "https://www.roughguides.com/united-arab-emirates/dubai/",
        "https://www.cntraveler.com/destinations/dubai",
        "https://www.timeout.com/dubai",
        # ── Dubai-specific blogs & portals ─────────────────────────────────────
        "https://www.dubai.com/v/blog/",
        "https://www.visitdubai.com/en/articles",
        "https://dubai.platinumlist.net/guide/",
        "https://www.dubaibarcode.com/",
        "https://www.timeoutdubai.com/",
        "https://www.dubailife.com/blog",
        "https://expatwoman.com/dubai",
        "https://dubizzle.com/community/",
        # ── Budget & backpacker ────────────────────────────────────────────────
        "https://thebrokebackpacker.com/dubai-travel-guide/",
        "https://www.nerdnomads.com/?s=dubai",
        "https://www.theplanetd.com/?s=dubai",
        "https://www.adventurouskate.com/?s=dubai",
        "https://www.indietraveller.co/?s=dubai",
        "https://www.lostwithpurpose.com/?s=dubai",
        "https://thesavvybackpacker.com/?s=dubai",
        # ── Food & dining ──────────────────────────────────────────────────────
        "https://www.zomato.com/dubai",
        "https://www.migrationology.com/?s=dubai",
        "https://www.eatyourworld.com/?s=dubai",
        "https://www.foodtourist.com/?s=dubai",
        "https://www.whatsondubai.ae/food-and-drink",
        # ── Expat & digital nomad ──────────────────────────────────────────────
        "https://nomadlist.com/dubai",
        "https://www.goatsontheroad.com/?s=dubai",
        "https://expertvagabond.com/?s=dubai",
        "https://www.internations.org/dubai-expats",
        # ── Hidden gems & culture ──────────────────────────────────────────────
        "https://www.alseerkgroup.com/blog",
        "https://www.dubaiexplorer.com/",
        "https://www.arabianbusiness.com/travel",
        # ── Listicles & magazines ──────────────────────────────────────────────
        "https://www.tripsavvy.com/dubai-4687986",
        "https://www.tripzilla.com/tag/dubai",
        "https://www.worldpackers.com/articles/dubai",
        "https://www.danflyingsolo.com/?s=dubai",
        "https://youngadventuress.com/?s=dubai",
        "https://www.handluggageonly.co.uk/?s=dubai",
        "https://www.nomadicchica.com/?s=dubai",
        "https://www.mylifesamovie.com/?s=dubai",
        "https://www.helenabunntravel.com/?s=dubai",
        "https://www.bruised-passports.com/?s=dubai",
        "https://theplanetleft.com/?s=dubai",
        "https://www.journeyera.com/?s=dubai",
        "https://hellosomewhere.com/?s=dubai",
    ],
}

# ── Metadata Tag Keywords ─────────────────────────────────────────────────────
# Used by metadata_builder.py for rule-based auto-tagging
BUDGET_KEYWORDS = {
    "free":   ["free", "no cost", "gratis", "zero cost"],
    "budget": ["budget", "cheap", "backpacker", "affordable", "shoestring", "hostel"],
    "mid":    ["mid-range", "boutique", "3-star", "4-star", "comfortable"],
    "luxury": ["luxury", "5-star", "resort", "high-end", "exclusive", "premium"],
}

HIDDEN_GEM_KEYWORDS = [
    "hidden gem", "off the beaten", "secret", "locals only", "tourist trap avoid",
    "underrated", "unknown", "crowd-free", "no tourists", "overlooked", "undiscovered",
]

WARNING_KEYWORDS = [
    "scam", "avoid", "dangerous", "unsafe", "warning", "rip off", "beware",
    "don't go", "overpriced", "closed", "flood", "monsoon problem",
]

WEATHER_KEYWORDS = {
    "dry_season":    ["dry season", "april to october", "best time", "sunny"],
    "avoid_monsoon": ["monsoon", "rainy season", "november", "december", "wet season", "flood"],
    "summer_heat":   ["summer heat", "june july august", "too hot", "extreme heat"],
}

FAMILY_KEYWORDS    = ["family", "children", "kids", "toddler", "baby", "family-friendly"]
SOLO_KEYWORDS      = ["solo", "alone", "backpacker", "digital nomad", "single traveller"]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
