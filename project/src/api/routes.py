"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, SearchRequest, SearchResponse
from src.services.search_service import SearchService

router = APIRouter(tags=["SemDatasheet"])


def get_search_service(request: Request) -> SearchService:
    """Resolve the shared search service from the application state."""
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("Search service is not initialized")
    return service


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search datasheet values by natural language query."""
    data = service.search(query=payload.query, top_k=payload.top_k)
    return SearchResponse(**data)


@router.get("/health", response_model=HealthResponse)
async def health(
    service: SearchService = Depends(get_search_service),
) -> HealthResponse:
    """Health endpoint with index/model state."""
    return HealthResponse(**service.health())


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
