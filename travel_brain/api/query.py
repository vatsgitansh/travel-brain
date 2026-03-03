"""
query.py — FastAPI router for travel knowledge queries.

POST /query  →  Natural language → Vector DB search → Ranked travel results
GET  /query/examples  →  Pre-built example queries for quick testing
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from travel_brain import config
from travel_brain.api.schemas import TravelQuery, TravelQueryResponse, TravelResult
from travel_brain.processing.embedder import embed_texts
from travel_brain.vectordb.base_client import VectorDBClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


# ── DB singleton (reused across requests) ─────────────────────────────────────

_db: Optional[VectorDBClient] = None


def get_db() -> VectorDBClient:
    """Return a cached vector DB client (lazy-initialized on first request)."""
    global _db
    if _db is None:
        if config.VECTOR_DB_PROVIDER == "pinecone":
            from travel_brain.vectordb.pinecone_client import PineconeClient
            _db = PineconeClient()
        else:
            from travel_brain.vectordb.chroma_client import ChromaClient
            _db = ChromaClient()
        logger.info(f"Vector DB client initialized: {config.VECTOR_DB_PROVIDER}")
    return _db


# ── Metadata Filter Builder ───────────────────────────────────────────────────

def build_filter(req: TravelQuery) -> Optional[dict]:
    """
    Convert the user's filter preferences into a vector DB filter dict.

    For Pinecone: {"location": {"$eq": "bali"}, "is_hidden_gem": {"$eq": True}}
    For ChromaDB: {"location": "bali", "is_hidden_gem": True}
    """
    filters: dict[str, Any] = {}

    if req.location:
        filters["location"] = req.location
    if req.budget_level:
        filters["budget_level"] = req.budget_level
    if req.hidden_gems_only:
        filters["is_hidden_gem"] = True
    if req.exclude_warnings:
        # ChromaDB uses simple equality; Pinecone uses operators
        # We handle this post-retrieval for maximum compatibility
        pass
    if req.family_friendly is not None:
        filters["is_family_friendly"] = req.family_friendly
    if req.weather_dependency:
        filters["weather_dependency"] = req.weather_dependency

    return filters if filters else None


def chroma_filter(f: dict) -> Optional[dict]:
    """
    Convert a flat equality filter dict to ChromaDB's where-clause format.

    ChromaDB quirks:
    - Single condition: {"key": "value"}  ← works fine
    - Multiple conditions: {"$and": [{"k1": "v1"}, {"k2": "v2"}]}  ← REQUIRED
    - Without $and, ChromaDB raises: 'Expected where to have exactly one operator'
    """
    if not f:
        return None
    items = list(f.items())
    if len(items) == 1:
        return {items[0][0]: items[0][1]}
    # Wrap multiple conditions in $and
    return {"$and": [{k: v} for k, v in items]}


def pinecone_filter(f: dict) -> dict:
    """Convert a simple equality filter dict to Pinecone operator syntax."""
    pinecone_f: dict[str, Any] = {}
    for k, v in f.items():
        pinecone_f[k] = {"$eq": v}
    return pinecone_f


# ── Format Results ────────────────────────────────────────────────────────────

def format_result(rank: int, raw: dict) -> TravelResult:
    """Convert a raw vector DB match into a structured TravelResult."""
    meta = raw.get("metadata", {})

    # Parse tags CSV → list
    tags_raw = meta.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    return TravelResult(
        rank=rank,
        score=round(float(raw.get("score", 0.0)), 4),
        text=raw.get("text", ""),
        source_type=meta.get("source_type", "unknown"),
        source_url=meta.get("source_url", ""),
        source_title=meta.get("source_title", ""),
        location=meta.get("location", ""),
        region=meta.get("region") or None,
        budget_level=meta.get("budget_level") or None,
        is_hidden_gem=meta.get("is_hidden_gem"),
        sentiment=meta.get("sentiment") or None,
        tags=tags,
        content_freshness=meta.get("content_freshness") or None,
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=TravelQueryResponse,
    summary="Query the Travel Brain",
    description="""
Search the Travel Brain's vector knowledge base with a natural language question.

**Examples:**
- `"What are hidden gems in Bali that locals love?"`
- `"Free things to do in Dubai without spending money"`
- `"Best beaches in Bali during dry season for families"`
- `"Scams to avoid in Dubai"`
    """,
)
async def query_travel_brain(req: TravelQuery) -> TravelQueryResponse:
    # 1. Embed the query
    try:
        query_vectors = embed_texts([req.query])
        query_vec = query_vectors[0]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}")

    # 2. Build metadata filter (excluding location — handled via namespace)
    raw_filter: dict[str, Any] = {}
    if req.budget_level:
        raw_filter["budget_level"] = req.budget_level
    if req.hidden_gems_only:
        raw_filter["is_hidden_gem"] = True
    if req.family_friendly is not None:
        raw_filter["is_family_friendly"] = req.family_friendly
    if req.weather_dependency:
        raw_filter["weather_dependency"] = req.weather_dependency
    # Note: exclude_warnings handled post-retrieval (avoids filter complexity)

    # 3. Convert filter to correct DB format
    db = get_db()
    db_filter: Optional[dict] = None
    if raw_filter:
        if config.VECTOR_DB_PROVIDER == "pinecone":
            db_filter = pinecone_filter(raw_filter)
        else:
            db_filter = chroma_filter(raw_filter)

    # 4. Namespace: use location as namespace for scoped search
    #    If no location specified, query all known namespaces and merge results
    namespace = req.location or ""
    fetch_k = req.top_k * 2 if req.exclude_warnings else req.top_k

    try:
        if namespace:
            # Scoped query — fast path
            raw_results = db.query(
                query_embedding=query_vec,
                top_k=fetch_k,
                namespace=namespace,
                filter=db_filter,
            )
        else:
            # No location filter — query each known namespace and merge
            all_results: list[dict] = []
            for loc in config.LOCATIONS:
                try:
                    loc_results = db.query(
                        query_embedding=query_vec,
                        top_k=fetch_k,
                        namespace=loc,
                        filter=db_filter,
                    )
                    all_results.extend(loc_results)
                except Exception:
                    pass  # Empty namespace is fine
            # Re-rank by score descending
            raw_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
    except Exception as e:
        logger.error(f"Vector DB query failed: {e}")
        raise HTTPException(status_code=503, detail=f"Vector DB error: {e}")

    # 6. Post-filter: exclude warning sentiment if requested
    if req.exclude_warnings:
        raw_results = [
            r for r in raw_results
            if r.get("metadata", {}).get("sentiment") != "warning"
        ]

    # 7. Cap at top_k
    raw_results = raw_results[:req.top_k]

    # 8. Format response
    results = [format_result(i + 1, r) for i, r in enumerate(raw_results)]

    return TravelQueryResponse(
        query=req.query,
        location_filter=req.location,
        results=results,
        result_count=len(results),
        vector_db=config.VECTOR_DB_PROVIDER,
        embedding_model=config.EMBEDDING_PROVIDER,
    )


# ── GET /query/examples ───────────────────────────────────────────────────────

@router.get(
    "/examples",
    summary="Example queries to test the Travel Brain",
)
async def get_example_queries() -> dict:
    return {
        "examples": [
            {
                "name": "Bali hidden gems",
                "request": {
                    "query": "What are the hidden gems in Bali that tourists don't know about?",
                    "location": "bali",
                    "hidden_gems_only": True,
                    "top_k": 5,
                },
            },
            {
                "name": "Free Dubai activities",
                "request": {
                    "query": "Free things to do in Dubai without spending money",
                    "location": "dubai",
                    "budget_level": "free",
                    "top_k": 5,
                },
            },
            {
                "name": "Bali family beaches dry season",
                "request": {
                    "query": "Best beaches in Bali for families during dry season",
                    "location": "bali",
                    "family_friendly": True,
                    "weather_dependency": "dry_season",
                    "top_k": 5,
                },
            },
            {
                "name": "Dubai scam warnings",
                "request": {
                    "query": "Scams and rip-offs to watch out for in Dubai",
                    "location": "dubai",
                    "exclude_warnings": False,
                    "top_k": 5,
                },
            },
            {
                "name": "Budget Bali travel tips",
                "request": {
                    "query": "How to travel Bali on a tight budget like a backpacker",
                    "location": "bali",
                    "budget_level": "budget",
                    "top_k": 8,
                },
            },
        ]
    }
