"""FAISS vector index with sentence-transformers and a deterministic fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from src.core.logging import get_logger
from src.data.models import Chunk
from src.indexing.bm25_index import tokenize

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import guard for lightweight environments
    SentenceTransformer = None  # type: ignore[assignment]

LOGGER = get_logger("vector_index")


class FallbackEncoder:
    """Deterministic hashing-based encoder used when ST is unavailable."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.backend = "fallback"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                index = int(digest, 16) % self.dimension
                embeddings[row, index] += 1.0
            norm = np.linalg.norm(embeddings[row])
            if norm > 0:
                embeddings[row] /= norm
        return embeddings


class SentenceTransformerEncoder:
    """Adapter around sentence-transformers with a graceful fallback."""

    def __init__(self, model_name: str, allow_model_download: bool = False) -> None:
        self.model_name = model_name
        self.allow_model_download = allow_model_download
        self.backend = "fallback"
        self.dimension = 384
        self.model = None

        if SentenceTransformer is None:
            self.model = FallbackEncoder(self.dimension)
            self.backend = "fallback"
            return

        try:
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                local_files_only=not allow_model_download,
            )
            self.backend = "sentence_transformers"
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "sentence_transformer_fallback",
                model_name=model_name,
                error=str(exc),
            )
            self.model = FallbackEncoder(self.dimension)
            self.backend = "fallback"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "sentence_transformers" and self.model is not None:
            vectors = self.model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        assert self.model is not None
        return self.model.encode(texts)


class VectorIndex:
    """A FAISS IP index over normalized embeddings."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: SentenceTransformerEncoder,
        index: faiss.Index,
        backend: str,
        model_name: str,
    ) -> None:
        self.chunks = list(chunks)
        self.encoder = encoder
        self.index = index
        self.backend = backend
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        model_name: str,
        allow_model_download: bool = False,
    ) -> "VectorIndex":
        encoder = SentenceTransformerEncoder(
            model_name=model_name,
            allow_model_download=allow_model_download,
        )
        index = faiss.IndexFlatIP(encoder.dimension)

        if chunks:
            passages = [f"passage: {chunk.text}" for chunk in chunks]
            vectors = encoder.encode(passages)
            vectors = np.ascontiguousarray(vectors.astype(np.float32))
            faiss.normalize_L2(vectors)
            index.add(vectors)

        return cls(
            chunks=chunks,
            encoder=encoder,
            index=index,
            backend=encoder.backend,
            model_name=model_name,
        )

    @classmethod
    def empty(
        cls,
        chunks: Sequence[Chunk],
        model_name: str,
        allow_model_download: bool = False,
    ) -> "VectorIndex":
        return cls.build(chunks=chunks, model_name=model_name, allow_model_download=allow_model_download)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return chunk ids and similarity scores."""
        if self.index.ntotal == 0:
            return []

        query_text = f"query: {query}"
        vector = self.encoder.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(vector)

        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, limit)

        results: List[Tuple[int, float]] = []
        for index_position, score in zip(indices[0], scores[0]):
            if index_position < 0:
                continue
            results.append((self.chunks[index_position].chunk_id, float(score)))
        return results

    def save(self, index_path: Path, meta_path: Path) -> None:
        """Persist FAISS index and metadata."""
        faiss.write_index(self.index, str(index_path))
        meta = {
            "backend": self.backend,
            "model_name": self.model_name,
            "dimension": self.encoder.dimension,
            "chunk_count": len(self.chunks),
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        chunks: Sequence[Chunk],
        index_path: Path,
        meta_path: Path,
        allow_model_download: bool = False,
    ) -> "VectorIndex":
        """Load FAISS index and reconstruct the encoder."""
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        index = faiss.read_index(str(index_path))
        encoder = SentenceTransformerEncoder(
            model_name=meta["model_name"],
            allow_model_download=allow_model_download,
        )
        return cls(
            chunks=chunks,
            encoder=encoder,
            index=index,
            backend=encoder.backend,
            model_name=meta["model_name"],
        )

    @property
    def model_loaded(self) -> bool:
        return self.backend == "sentence_transformers"
