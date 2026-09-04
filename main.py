"""Fake News Detector entry point.

Runs the FastAPI application with uvicorn. Configuration is read from
environment variables (see ``app/config.py`` and ``.env.example``).
"""

import logging
import os

import uvicorn

from app.config import settings
from app.main import app

LOGGING_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOGGING_FORMAT)
    # Keep uvicorn's own access logs readable.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


if __name__ == "__main__":
    _configure_logging()
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
