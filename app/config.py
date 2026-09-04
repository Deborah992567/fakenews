"""Application configuration.

Configuration values are read from environment variables with sensible
defaults, and may be overridden at runtime through an optional ``.env`` file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    """Typed access to application settings.

    All values are resolved at import time.  If a ``.env`` file exists in the
    project root its contents are loaded first; existing environment variables
    always win.
    """

    HOST: str = _env_str("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    MODEL_PATH: str = _env_str("MODEL_PATH", "my_model.h5")
    VECTORIZER_PATH: str = _env_str("VECTORIZER_PATH", "countvectorizer.pkl")
    UNCERTAINTY_THRESHOLD: float = _env_float("UNCERTAINTY_THRESHOLD", 0.10)
    MAX_INPUT_LENGTH: int = _env_int("MAX_INPUT_LENGTH", 20_000)
    MAX_URL_RESPONSE_SIZE: int = _env_int("MAX_URL_RESPONSE_SIZE", 1_000_000)
    REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 10)
    MAX_REDIRECTS: int = _env_int("MAX_REDIRECTS", 5)
    CORS_ORIGINS: str = _env_str("CORS_ORIGINS", "*")
    TOP_FEATURES: int = _env_int("TOP_FEATURES", 10)

    @property
    def model_file(self) -> Path:
        path = Path(self.MODEL_PATH)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def vectorizer_file(self) -> Path:
        path = Path(self.VECTORIZER_PATH)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def allowed_origins(self) -> list[str]:
        raw = self.CORS_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def summary(self) -> dict[str, object]:
        """Return a serialisable summary of effective settings (no secrets)."""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "model_path": str(self.model_file),
            "vectorizer_path": str(self.vectorizer_file),
            "uncertainty_threshold": self.UNCERTAINTY_THRESHOLD,
            "max_input_length": self.MAX_INPUT_LENGTH,
            "top_features": self.TOP_FEATURES,
            "request_timeout": self.REQUEST_TIMEOUT,
            "max_redirects": self.MAX_REDIRECTS,
        }


settings = Settings()