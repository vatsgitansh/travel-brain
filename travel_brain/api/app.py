"""
app.py — Travel Brain FastAPI application.

Start the server:
  uvicorn travel_brain.api.app:app --reload --port 8000

Then open:
  http://localhost:8000/docs   →  Swagger UI (interactive API explorer)
  http://localhost:8000/redoc  →  ReDoc documentation
"""

import logging
import sys
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
from travel_brain.api.itinerary import router as itinerary_router

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

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
    version="1.0.0",
    contact={
        "name": "Travel Brain CTO",
        "email": "cto@travelbrain.ai",
    },
    license_info={
        "name": "Proprietary",
    },
)

# ── CORS (allow all origins for development) ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query_router)
app.include_router(chat_router)
app.include_router(advisories_router)
app.include_router(itinerary_router)

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
    """Returns the API status and vector DB statistics."""
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
        "vector_db": config.VECTOR_DB_PROVIDER,
        "embedding_model": config.EMBEDDING_PROVIDER,
        "embedding_dim": config.EMBEDDING_DIM,
        "locations": config.LOCATIONS,
        "db_stats": db_stats,
        "version": "1.0.0",
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
