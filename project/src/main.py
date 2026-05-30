"""Application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.services.search_service import SearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = SearchService(resolved_settings)
        service.initialize()
        app.state.search_service = service
        yield

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": resolved_settings.app_name, "version": resolved_settings.app_version}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
