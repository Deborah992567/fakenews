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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Typed access to application settings.

    All values are resolved at import time.  If a ``.env`` file exists in the
    project root its contents are loaded first; existing environment variables
    always win.
    """

    HOST: str = _env_str("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    MODEL_PATH: str = _env_str("MODEL_PATH", "my_model_lr.pkl")
    VECTORIZER_PATH: str = _env_str("VECTORIZER_PATH", "my_tfidf_vectorizer.pkl")
    UNCERTAINTY_THRESHOLD: float = _env_float("UNCERTAINTY_THRESHOLD", 0.10)
    MAX_INPUT_LENGTH: int = _env_int("MAX_INPUT_LENGTH", 20_000)
    MAX_URL_LENGTH: int = _env_int("MAX_URL_LENGTH", 2048)
    MAX_REQUEST_BODY_BYTES: int = _env_int("MAX_REQUEST_BODY_BYTES", 300_000)
    MAX_URL_RESPONSE_SIZE: int = _env_int("MAX_URL_RESPONSE_SIZE", 1_000_000)
    REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 10)
    CONNECT_TIMEOUT: float = _env_float("CONNECT_TIMEOUT", 5)
    MAX_REDIRECTS: int = _env_int("MAX_REDIRECTS", 5)
    CORS_ORIGINS: str = _env_str("CORS_ORIGINS", "*")
    TOP_FEATURES: int = _env_int("TOP_FEATURES", 10)
    CACHE_URL_ENABLED: bool = _env_bool("CACHE_URL_ENABLED", True)
    CACHE_URL_TTL_SECONDS: int = _env_int("CACHE_URL_TTL_SECONDS", 600)
    CACHE_URL_MAX_ITEMS: int = _env_int("CACHE_URL_MAX_ITEMS", 512)
    RATE_LIMIT_ENABLED: bool = _env_bool("RATE_LIMIT_ENABLED", True)
    RATE_LIMIT_REQUESTS: int = _env_int("RATE_LIMIT_REQUESTS", 120)
    RATE_LIMIT_WINDOW_SECONDS: int = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    RATE_LIMIT_MAX_IPS: int = _env_int("RATE_LIMIT_MAX_IPS", 10_000)

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
            "max_url_length": self.MAX_URL_LENGTH,
            "max_request_body_bytes": self.MAX_REQUEST_BODY_BYTES,
            "top_features": self.TOP_FEATURES,
            "rate_limit_enabled": self.RATE_LIMIT_ENABLED,
            "rate_limit": f"{self.RATE_LIMIT_REQUESTS}/{self.RATE_LIMIT_WINDOW_SECONDS}s",
            "request_timeout": self.REQUEST_TIMEOUT,
            "connect_timeout": self.CONNECT_TIMEOUT,
            "max_redirects": self.MAX_REDIRECTS,
        }

    def validate(self) -> list[str]:
        """Return a list of warning strings for obviously wrong config values."""
        warnings: list[str] = []
        if not 0.0 < self.UNCERTAINTY_THRESHOLD < 1.0:
            warnings.append(
                f"UNCERTAINTY_THRESHOLD={self.UNCERTAINTY_THRESHOLD} is outside (0,1); "
                "uncertain predictions may never trigger."
            )
        if self.MAX_INPUT_LENGTH < 100:
            warnings.append(
                f"MAX_INPUT_LENGTH={self.MAX_INPUT_LENGTH} is very short; "
                "most articles will be rejected."
            )
        if self.MAX_URL_LENGTH < 20:
            warnings.append(
                f"MAX_URL_LENGTH={self.MAX_URL_LENGTH} is suspiciously short "
                "and will reject most valid article URLs."
            )
        if self.MAX_REQUEST_BODY_BYTES < 1024:
            warnings.append(
                f"MAX_REQUEST_BODY_BYTES={self.MAX_REQUEST_BODY_BYTES} is very "
                "small and will reject legitimate analysis requests."
            )
        if self.RATE_LIMIT_REQUESTS < 1:
            warnings.append(f"RATE_LIMIT_REQUESTS={self.RATE_LIMIT_REQUESTS} must be >= 1.")
        if self.RATE_LIMIT_WINDOW_SECONDS < 1:
            warnings.append(
                f"RATE_LIMIT_WINDOW_SECONDS={self.RATE_LIMIT_WINDOW_SECONDS} must be >= 1."
            )
        if self.CACHE_URL_MAX_ITEMS < 1:
            warnings.append(f"CACHE_URL_MAX_ITEMS={self.CACHE_URL_MAX_ITEMS} must be >= 1.")
        if self.CACHE_URL_TTL_SECONDS < 0:
            warnings.append(f"CACHE_URL_TTL_SECONDS={self.CACHE_URL_TTL_SECONDS} must be >= 0.")
        if self.PORT < 1 or self.PORT > 65535:
            warnings.append(f"PORT={self.PORT} is outside valid range 1-65535.")
        if self.TOP_FEATURES < 1:
            warnings.append(f"TOP_FEATURES={self.TOP_FEATURES} must be >= 1.")
        return warnings


settings = Settings()