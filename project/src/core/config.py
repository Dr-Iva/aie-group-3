"""Application configuration loaded from .env and config.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = BASE_DIR / ".env"
DEFAULT_CONFIG_FILE = BASE_DIR / "config.yaml"


class Settings(BaseSettings):
    """Runtime settings for the service."""

    model_config = SettingsConfigDict(extra="ignore")

    app_name: str = Field(default="SemDatasheet")
    app_version: str = Field(default="1.0.0")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw")
    artifacts_dir: Path = Field(default=BASE_DIR / "artifacts")

    chunk_size: int = Field(default=300, ge=200, le=400)
    chunk_overlap: int = Field(default=50, ge=0, le=150)

    embedding_model_name: str = Field(default="intfloat/multilingual-e5-small")
    allow_model_download: bool = Field(default=False)

    hybrid_alpha: float = Field(default=0.65, ge=0.0, le=1.0)
    default_top_k: int = Field(default=3, ge=1)
    max_top_k: int = Field(default=10, ge=1)

    log_level: str = Field(default="INFO")

    mlflow_tracking_uri: str = Field(default=f"file://{BASE_DIR / 'artifacts' / 'mlruns'}")
    mlflow_experiment_name: str = Field(default="SemDatasheet")

    @field_validator("raw_data_dir", "artifacts_dir")
    @classmethod
    def _expand_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper().strip()


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must contain a mapping, got {type(data)!r}")

    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        normalized[str(key).lower()] = value
    return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from .env and config.yaml with env precedence."""
    load_dotenv(DEFAULT_ENV_FILE, override=False)

    yaml_data = _load_yaml_config(DEFAULT_CONFIG_FILE)

    env_overrides: Dict[str, Any] = {}
    for field_name in Settings.model_fields:
        env_value = os.getenv(field_name.upper())
        if env_value is not None:
            env_overrides[field_name] = env_value

    merged = {**yaml_data, **env_overrides}
    settings = Settings(**merged)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "mlruns").mkdir(parents=True, exist_ok=True)

    return settings
