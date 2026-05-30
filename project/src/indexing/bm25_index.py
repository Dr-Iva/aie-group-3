"""BM25 index based on rank_bm25."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.models import Chunk

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9%°]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 and fallback embeddings."""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


class BM25Index:
    """Thin wrapper around BM25Okapi with chunk metadata."""

    def __init__(self, chunks: Sequence[Chunk], corpus_tokens: Sequence[Sequence[str]]) -> None:
        self.chunks = list(chunks)
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "BM25Index":
        corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        return cls(chunks=chunks, corpus_tokens=corpus_tokens)

    @classmethod
    def empty(cls) -> "BM25Index":
        return cls(chunks=[], corpus_tokens=[])

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return chunk ids and BM25 scores."""
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        limit = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:limit]

        results: List[Tuple[int, float]] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append((self.chunks[index].chunk_id, score))
        return results

    def save(self, path: Path) -> None:
        """Persist the index state to disk."""
        payload = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "corpus_tokens": self.corpus_tokens,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load the BM25 state from disk."""
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        corpus_tokens = payload["corpus_tokens"]
        return cls(chunks=chunks, corpus_tokens=corpus_tokens)
