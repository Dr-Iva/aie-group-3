"""Index builder service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.core.config import Settings
from src.core.logging import get_logger
from src.core.metrics import INDEX_SIZE
from src.data.chunker import PDFChunker
from src.data.loader import PDFLoader
from src.data.models import Chunk, LoadedDocument
from src.indexing.artifact_store import ArtifactStore
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex

try:
    import mlflow
except Exception:  # pragma: no cover - optional runtime dependency guard
    mlflow = None  # type: ignore[assignment]


@dataclass(slots=True)
class BuildResult:
    """Summary of a full indexing run."""

    documents: int
    chunks: int
    model_loaded: bool


class IndexBuilder:
    """Build BM25 and FAISS indexes from raw PDFs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger("index_builder")
        self.loader = PDFLoader(settings.raw_data_dir)
        self.chunker = PDFChunker(settings.chunk_size, settings.chunk_overlap)
        self.store = ArtifactStore(settings.artifacts_dir)

    def build(self) -> BuildResult:
        """Load PDFs, chunk them, build indices and persist artifacts."""
        documents = self.loader.load_documents()
        chunks = self._chunk_documents(documents)

        bm25_index = BM25Index.from_chunks(chunks) if chunks else BM25Index.empty()
        vector_index = VectorIndex.build(
            chunks=chunks,
            model_name=self.settings.embedding_model_name,
            allow_model_download=self.settings.allow_model_download,
        )

        self.store.save_chunks(chunks)
        bm25_index.save(self.store.bm25_path)
        vector_index.save(self.store.vector_index_path, self.store.vector_meta_path)

        manifest = {
            "app_name": self.settings.app_name,
            "app_version": self.settings.app_version,
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "embedding_model_name": self.settings.embedding_model_name,
            "hybrid_alpha": self.settings.hybrid_alpha,
            "default_top_k": self.settings.default_top_k,
            "model_loaded": vector_index.model_loaded,
        }
        self.store.save_manifest(manifest)

        self._log_mlflow(len(documents), len(chunks))
        INDEX_SIZE.set(len(chunks))

        self.logger.info(
            "index_build_complete",
            documents=len(documents),
            chunks=len(chunks),
            model_loaded=vector_index.model_loaded,
        )
        return BuildResult(
            documents=len(documents),
            chunks=len(chunks),
            model_loaded=vector_index.model_loaded,
        )

    def _chunk_documents(self, documents: List[LoadedDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []
        next_chunk_id = 0

        for document in documents:
            for page in document.pages:
                page_chunks, next_chunk_id = self.chunker.chunk_page(
                    page=page,
                    start_chunk_id=next_chunk_id,
                )
                chunks.extend(page_chunks)

        return chunks

    def _log_mlflow(self, documents: int, chunks: int) -> None:
        if mlflow is None:
            return

        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            with mlflow.start_run(run_name="index_build"):
                mlflow.log_param("alpha", self.settings.hybrid_alpha)
                mlflow.log_param("top_k", self.settings.default_top_k)
                mlflow.log_param("documents", documents)
                mlflow.log_metric("chunks", chunks)
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            self.logger.warning("mlflow_logging_failed", error=str(exc))
