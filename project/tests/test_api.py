from __future__ import annotations

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from src.core.config import Settings
from src.main import create_app


def _create_sample_pdf(pdf_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Electrical Characteristics\nSupply voltage: 1.8V to 6.0V.\nTypical Performance Characteristics\nGain is stable.",
        fontsize=12,
    )
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Absolute Maximum Ratings\nInput voltage should not exceed 6.0V.",
        fontsize=12,
    )
    doc.save(str(pdf_path))
    doc.close()


def test_search_and_health_endpoints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _create_sample_pdf(raw_dir / "MCP6002_datasheet.pdf")

    settings = Settings(
        app_name="SemDatasheet",
        app_version="1.0.0",
        raw_data_dir=raw_dir,
        artifacts_dir=artifacts_dir,
        chunk_size=300,
        chunk_overlap=50,
        embedding_model_name="intfloat/multilingual-e5-small",
        allow_model_download=False,
        hybrid_alpha=0.65,
        default_top_k=3,
        max_top_k=5,
        log_level="INFO",
        mlflow_tracking_uri=f"file://{artifacts_dir / 'mlruns'}",
        mlflow_experiment_name="SemDatasheet",
    )

    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["indexed_chunks"] >= 1

        response = client.post(
            "/search",
            json={"query": "Максимальное напряжение питания MCP6002", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processing_time_ms"] >= 0
        assert len(data["results"]) >= 1
        assert data["results"][0]["document"] == "MCP6002_datasheet.pdf"
        assert "V" in str(data["results"][0]["extracted_value"])
