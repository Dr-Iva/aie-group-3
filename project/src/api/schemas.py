"""Pydantic schemas for the public API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class SearchRequest(BaseModel):
    """Search request payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(default=3, ge=1, description="Number of results to return")


class SearchItem(BaseModel):
    """Single search result item."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1)
    document: str
    page: int = Field(..., ge=1)
    text_snippet: str
    extracted_value: Optional[str] = None
    unit: Optional[str] = None
    score: float = Field(..., ge=0.0)


class SearchResponse(BaseModel):
    """Search response payload."""

    model_config = ConfigDict(extra="forbid")

    results: List[SearchItem] = Field(default_factory=list)
    processing_time_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    model_loaded: bool
    index_loaded: bool
    indexed_chunks: int = Field(..., ge=0)
