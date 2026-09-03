"""Fake News Detector — FastAPI application entry point.

The app loads the trained model and vectorizer during application startup
(lifespan), validates inputs through Pydantic, serves a static frontend from
``/``, and exposes a shared prediction pipeline used by both the pasted-text
and URL endpoints.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.model import ModelLoadError, ModelService
from app.scraper import ScrapeError, fetch_article_text
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    UrlRequest,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class AppState:
    """Shared mutable state stored on the FastAPI application."""

    def __init__(self) -> None:
        self.model: ModelService | None = None


state = AppState()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model/vectorizer once during startup and store in state.
        service = ModelService(settings.model_file, settings.vectorizer_file)
        try:
            service.load()
        except ModelLoadError as exc:
            # Do not silently continue; the app is unusable for predictions.
            state.model = None
            raise RuntimeError(str(exc)) from exc
        state.model = service
        yield
        state.model = None

    application = FastAPI(
        title="Fake News Detector",
        description=(
            "Analyse whether a piece of news text (pasted or at a URL) is "
            "likely to be real or fake, including explainability and "
            "uncertainty handling."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve the static frontend.
    if FRONTEND_DIR.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=FRONTEND_DIR),
            name="static",
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=bool(state.model and state.model.is_loaded),
            vectorizer_loaded=bool(state.model and state.model.is_loaded),
        )

    def _require_model() -> ModelService:
        if state.model is None or not state.model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="The detector model is not loaded. Please try again later.",
            )
        return state.model

    def _to_response(service: ModelService, raw_text: str, source_type: str) -> PredictResponse:
        prediction = service.predict(raw_text)
        return PredictResponse(
            label=prediction.label,
            confidence=prediction.confidence,
            probability_real=round(prediction.probability_real * 100.0, 2),
            probability_fake=round(prediction.probability_fake * 100.0, 2),
            explanation={
                "top_influential_words": [
                    {
                        "word": item.word,
                        "impact": item.impact,
                        "direction": item.direction,
                    }
                    for item in prediction.explanation
                ]
            }
            if prediction.explanation
            else None,
            source_type=source_type,
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        service = _require_model()
        return _to_response(service, req.news, "text")

    @application.post("/predict-url", response_model=PredictResponse)
    def predict_url(req: UrlRequest) -> PredictResponse:
        service = _require_model()
        try:
            article_text = fetch_article_text(req.url)
        except ScrapeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_response(service, article_text, "url")

    return application


app = create_app()
