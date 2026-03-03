"""
schemas.py — Pydantic request/response models for the Travel Brain Query API.

These define the exact shape of input and output for every endpoint.
The CEO / product team can use these as the API contract spec.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Query Request ─────────────────────────────────────────────────────────────

class TravelQuery(BaseModel):
    """
    The request body for a travel knowledge query.
    All filter fields are optional — the more you provide, the more targeted
    the results.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language travel question",
        examples=["What are the best hidden beaches in Bali during dry season?"],
    )
    location: Optional[Literal["bali", "dubai"]] = Field(
        default=None,
        description="Filter results to a specific destination",
    )
    budget_level: Optional[Literal["free", "budget", "mid", "luxury"]] = Field(
        default=None,
        description="Filter by budget tier",
    )
    hidden_gems_only: bool = Field(
        default=False,
        description="If True, only return results tagged as hidden gems",
    )
    exclude_warnings: bool = Field(
        default=False,
        description="If True, exclude results with 'warning' sentiment (scams, closures etc.)",
    )
    family_friendly: Optional[bool] = Field(
        default=None,
        description="Filter for family-friendly spots (True), adults-only (False), or no filter (None)",
    )
    weather_dependency: Optional[Literal["dry_season", "avoid_monsoon", "summer_heat", "any"]] = Field(
        default=None,
        description="Filter by weather suitability",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1–20)",
    )


# ── Individual Result ─────────────────────────────────────────────────────────

class TravelResult(BaseModel):
    """A single retrieved chunk from the vector DB, with its relevance score and metadata."""

    rank:             int
    score:            float = Field(description="Cosine similarity score (0–1, higher = more relevant)")
    text:             str   = Field(description="The actual travel content")
    source_type:      str   = Field(description="youtube | reddit | blog")
    source_url:       str
    source_title:     str
    location:         str
    region:           Optional[str]   = None
    budget_level:     Optional[str]   = None
    is_hidden_gem:    Optional[bool]  = None
    sentiment:        Optional[str]   = None
    tags:             list[str]       = Field(default_factory=list)
    content_freshness: Optional[str] = None


# ── Query Response ────────────────────────────────────────────────────────────

class TravelQueryResponse(BaseModel):
    """The full API response for a travel query."""

    query:          str
    location_filter: Optional[str]
    results:        list[TravelResult]
    result_count:   int
    vector_db:      str = Field(description="Which vector DB was used (pinecone | chroma)")
    embedding_model: str


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:     Literal["ok", "degraded"]
    vector_db:  str
    db_stats:   dict
    version:    str = "1.0.0"


# ── Ingest Trigger (optional, for manual re-runs via API) ─────────────────────

class IngestRequest(BaseModel):
    location: Literal["bali", "dubai", "all"] = "all"
    source:   Literal["youtube", "reddit", "blog", "all"] = "all"
    limit:    int = Field(default=20, ge=1, le=200)
    dry_run:  bool = True   # Default to dry_run=True for safety via API


class IngestResponse(BaseModel):
    status:         str
    location:       str
    source:         str
    dry_run:        bool
    message:        str
