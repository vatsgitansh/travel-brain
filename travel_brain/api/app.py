"""
app.py — Travel Brain FastAPI application.

Start the server:
  uvicorn travel_brain.api.app:app --reload --port 8000

Then open:
  http://localhost:8000/ui    →  Chat UI
  http://localhost:8000/docs  →  Swagger UI
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config
from travel_brain.api.query import router as query_router
from travel_brain.api.chat  import router as chat_router
from travel_brain.api.advisories import router as advisories_router

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")


# ── Startup: Pre-warm models to eliminate cold-start lag on first request ─────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the embedding model and ChromaDB client on server startup."""
    logger.info("🔥 Pre-warming embedding model and vector DB client...")
    try:
        from travel_brain.processing.embedder import embed_texts
        from travel_brain.api.query import get_db
        embed_texts(["warmup"])   # Forces sentence-transformers to load into memory
        get_db()                  # Opens the ChromaDB persistent client
        logger.info("✅ Model and DB warm-up complete. First request will be instant.")
    except Exception as e:
        logger.warning(f"⚠️  Warm-up failed (non-fatal): {e}")
    yield
    logger.info("Travel Brain API shutting down.")


# ── App Definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="🧠 Travel Brain API",
    description="""
## The proprietary travel intelligence API

Powered by a vector database of scraped travel knowledge from:
- 🎬 **YouTube** travel vlogs and transcripts  
- 💬 **Reddit** posts from r/bali, r/dubai, r/travel, r/shoestring  
- 📝 **Niche travel blogs** (off-the-beaten-path content)

### Features
- **Natural language search** — ask anything about Bali or Dubai
- **Smart metadata filters** — filter by budget, hidden gems, weather, family-friendly
- **Source provenance** — every result links back to its original source
- **Warning detection** — surface scam alerts and travel warnings

### Current Coverage
- 🌴 **Bali** — hidden beaches, rice terraces, warung food, scooter routes, digital nomad spots
- 🏙️ **Dubai** — free activities, Old Dubai, desert experiences, budget tips
    """,
    version="1.1.0",
    contact={
        "name": "Travel Brain CTO",
        "email": "cto@travelbrain.ai",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
)

# ── CORS — Locked to configured origins ──────────────────────────────────────
# In production, set CORS_ORIGINS env var to your domain(s), e.g.:
#   CORS_ORIGINS=https://travelbrain.ai,https://app.travelbrain.ai
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query_router)
app.include_router(chat_router)
app.include_router(advisories_router)

# ── Chat UI ───────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/ui", include_in_schema=False)
async def chat_ui() -> FileResponse:
    """Serve the Travel Brain Chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
async def health_check() -> dict:
    """Returns API status, vector DB statistics, and configuration details."""
    from travel_brain.api.query import get_db
    try:
        db = get_db()
        db_stats = db.describe()
        status = "ok"
    except Exception as e:
        db_stats = {"error": str(e)}
        status = "degraded"

    return {
        "status": status,
        "version": "1.1.0",
        "vector_db": config.VECTOR_DB_PROVIDER,
        "embedding_model": config.EMBEDDING_PROVIDER,
        "embedding_dim": config.EMBEDDING_DIM,
        "locations": config.LOCATIONS,
        "db_stats": db_stats,
    }


@app.get("/", tags=["System"], include_in_schema=False)
async def root() -> dict:
    return {
        "message": "🧠 Travel Brain API is running",
        "chat_ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "examples": "GET /query/examples",
    }
