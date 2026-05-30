"""Persistence layer for index artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.models import Chunk


class ArtifactStore:
    """Store and load index artifacts under artifacts/."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.artifacts_dir / "chunks.json"

    @property
    def bm25_path(self) -> Path:
        return self.artifacts_dir / "bm25.pkl"

    @property
    def vector_index_path(self) -> Path:
        return self.artifacts_dir / "faiss.index"

    @property
    def vector_meta_path(self) -> Path:
        return self.artifacts_dir / "vector_meta.json"

    def exists(self) -> bool:
        return all(
            path.exists()
            for path in (
                self.manifest_path,
                self.chunks_path,
                self.bm25_path,
                self.vector_index_path,
                self.vector_meta_path,
            )
        )

    def save_chunks(self, chunks: List[Chunk]) -> None:
        payload = [chunk.to_dict() for chunk in chunks]
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_chunks(self) -> List[Chunk]:
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [Chunk.from_dict(item) for item in payload]

    def save_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_manifest(self, payload: Dict[str, Any]) -> None:
        self.save_json(self.manifest_path, payload)

    def load_manifest(self) -> Dict[str, Any]:
        return self.load_json(self.manifest_path)
