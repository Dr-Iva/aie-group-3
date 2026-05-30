# app/services/search_service.py
"""Search service orchestrating loading, indexing and query execution."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

from src.core.config import Settings
from src.core.logging import get_logger
from src.core.metrics import INDEX_SIZE, SEARCH_LATENCY_SECONDS, SEARCH_REQUESTS_TOTAL
from src.data.chunker import PDFChunker
from src.data.loader import PDFLoader
from src.data.models import Chunk
from src.extraction.value_extractor import ExtractedValue, ValueExtractor
from src.indexing.artifact_store import ArtifactStore
from src.indexing.bm25_index import BM25Index
from src.indexing.hybrid_search import HybridSearch
from src.indexing.vector_index import VectorIndex
from src.services.index_builder import IndexBuilder


class SearchService:
    """High-level service used by the API layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger("search_service")
        self.store = ArtifactStore(settings.artifacts_dir)
        self.value_extractor = ValueExtractor()
        self.lock = RLock()

        self.chunks: List[Chunk] = []
        self.bm25_index: BM25Index | None = None
        self.vector_index: VectorIndex | None = None
        self.hybrid_search: HybridSearch | None = None
        self.manifest: dict[str, Any] = {}
        self.initialized = False

    def initialize(self) -> None:
        """Load artifacts or rebuild them from raw PDFs."""
        with self.lock:
            if self.initialized:
                return

            if self.store.exists():
                try:
                    self._load_artifacts()
                    if not self._artifacts_match_settings(self.manifest):
                        self.logger.info("artifact_settings_mismatch_rebuild")
                        self._build_from_raw()
                        self._load_artifacts()
                except Exception as exc:
                    self.logger.warning("artifact_loading_failed_rebuild", error=str(exc))
                    self._build_from_raw()
                    self._load_artifacts()
            else:
                self._build_from_raw()
                self._load_artifacts()

            self._update_metrics()
            self.initialized = True

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search the indexed datasheets."""
        self.initialize()

        normalized_top_k = self._normalize_top_k(top_k)
        start_time = time.perf_counter()
        SEARCH_REQUESTS_TOTAL.inc()

        if self.hybrid_search is None:
            return {"results": [], "processing_time_ms": 0.0}

        hits = self.hybrid_search.search(query=query, top_k=normalized_top_k)

        results: List[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            extraction = self.value_extractor.extract(query=query, text=hit.chunk.text)
            snippet = self._build_snippet(hit.chunk.text, extraction)
            results.append(
                {
                    "rank": rank,
                    "document": hit.chunk.filename,
                    "page": hit.chunk.page,
                    "text_snippet": snippet,
                    "extracted_value": extraction.value if extraction else None,
                    "unit": extraction.unit if extraction else None,
                    "score": round(float(hit.score), 4),
                }
            )

        processing_time = time.perf_counter() - start_time
        SEARCH_LATENCY_SECONDS.observe(processing_time)

        self.logger.info(
            "search_completed",
            query=query,
            top_k=normalized_top_k,
            results=len(results),
            processing_time_ms=round(processing_time * 1000.0, 2),
        )

        return {
            "results": results,
            "processing_time_ms": round(processing_time * 1000.0, 2),
        }

    def health(self) -> dict[str, Any]:
        """Return health status required by the API contract."""
        self.initialize()
        return {
            "status": "ok" if self.index_loaded else "empty",
            "version": self.settings.app_version,
            "model_loaded": bool(self.vector_index and self.vector_index.model_loaded),
            "index_loaded": self.index_loaded,
            "indexed_chunks": len(self.chunks),
        }

    @property
    def index_loaded(self) -> bool:
        return self.bm25_index is not None and self.vector_index is not None

    def _build_from_raw(self) -> None:
        builder = IndexBuilder(self.settings)
        builder.build()

    def _load_artifacts(self) -> None:
        self.manifest = self.store.load_manifest()
        self.chunks = self.store.load_chunks()
        self.bm25_index = BM25Index.load(self.store.bm25_path)
        self.vector_index = VectorIndex.load(
            chunks=self.chunks,
            index_path=self.store.vector_index_path,
            meta_path=self.store.vector_meta_path,
            allow_model_download=self.settings.allow_model_download,
        )
        self.hybrid_search = HybridSearch(
            bm25_index=self.bm25_index,
            vector_index=self.vector_index,
            alpha=self.settings.hybrid_alpha,
        )

    def _update_metrics(self) -> None:
        INDEX_SIZE.set(len(self.chunks))

    def _normalize_top_k(self, top_k: int | None) -> int:
        value = top_k if top_k is not None else self.settings.default_top_k
        value = max(1, value)
        return min(value, self.settings.max_top_k)

    def _build_snippet(self, text: str, extraction: ExtractedValue | None, width: int = 220) -> str:
        if extraction is None:
            snippet = text[:width].strip()
            return snippet + ("..." if len(text) > width else "")

        start = max(0, extraction.start - 90)
        end = min(len(text), extraction.end + 130)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet

    def _artifacts_match_settings(self, manifest: dict[str, Any]) -> bool:
        if not manifest:
            return False

        return (
            manifest.get("embedding_model_name") == self.settings.embedding_model_name
            and int(manifest.get("chunk_size", -1)) == self.settings.chunk_size
            and int(manifest.get("chunk_overlap", -1)) == self.settings.chunk_overlap
            and float(manifest.get("hybrid_alpha", -1.0)) == self.settings.hybrid_alpha
        )
