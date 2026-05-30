"""Hybrid search combining BM25 and vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.data.models import Chunk, SearchHit
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex


class HybridSearch:
    """Combine lexical and semantic scores with a configurable alpha."""

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        alpha: float,
    ) -> None:
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.alpha = alpha
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in bm25_index.chunks or vector_index.chunks}

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """Run BM25 and vector search and merge the scores."""
        if not query.strip():
            return []

        bm25_hits = self.bm25_index.search(query, top_k=max(top_k * 4, top_k))
        vector_hits = self.vector_index.search(query, top_k=max(top_k * 4, top_k))

        bm25_scores = self._normalize_scores(dict(bm25_hits))
        vector_scores = self._normalize_scores(dict(vector_hits))

        all_ids = set(bm25_scores) | set(vector_scores)
        scored = []
        for chunk_id in all_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            vector_score = vector_scores.get(chunk_id, 0.0)
            hybrid_score = self.alpha * vector_score + (1.0 - self.alpha) * bm25_score

            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            scored.append(SearchHit(chunk=chunk, score=float(hybrid_score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}

        values = np.asarray(list(scores.values()), dtype=np.float32)
        min_score = float(values.min())
        max_score = float(values.max())

        if abs(max_score - min_score) < 1e-9:
            return {key: (1.0 if value > 0 else 0.0) for key, value in scores.items()}

        return {
            key: float((value - min_score) / (max_score - min_score))
            for key, value in scores.items()
        }
