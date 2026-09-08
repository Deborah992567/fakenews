"""Fake News Detector — FastAPI application entry point.

The app loads the trained model and vectorizer during application startup
(lifespan), validates inputs through Pydantic, serves a static frontend from
``/``, and exposes a shared prediction pipeline used by both the pasted-text
and URL endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.cache import TTLCache, cache_key
from app.config import settings
from app.model import ModelLoadError, ModelService
from app.observability import RequestIDMiddleware, attach_request_id_filter
from app.prediction_log import PredictionEntry, prediction_log
from app.preprocessing import ensure_stopwords_available
from app.ratelimit import RateLimitMiddleware, SlidingWindowRateLimiter
from app.scraper import ExtractResult, ScrapeError, fetch_article
from app.security import RequestBodyLimitMiddleware
from app.schemas import (
    HealthResponse,
    LiveResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    UrlRequest,
)

logger = logging.getLogger("fakenews")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class AppState:
    """Shared mutable state stored on the FastAPI application.

    Owned by ``create_app`` (``app.state.app_state``).  The module-level
    ``state`` below is bound to the default application so existing tests keep
    working, but every new ``create_app()`` call receives isolated state.
    """

    def __init__(self) -> None:
        self.model: ModelService | None = None
        # Bounded in-memory memoisation of fetched/extracted URL results. It
        # never stores request bodies; only deterministic extraction outputs.
        self.url_cache: TTLCache = TTLCache(
            ttl_seconds=settings.CACHE_URL_TTL_SECONDS,
            max_items=settings.CACHE_URL_MAX_ITEMS,
        )
        # Per-process sliding-window rate limiter (audit B5). Each application
        # instance owns its counters; see docs/concurrency.md for limits.
        self.rate_limiter: SlidingWindowRateLimiter = SlidingWindowRateLimiter(
            limit=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
            max_keys=settings.RATE_LIMIT_MAX_IPS,
        )


def create_app(app_state: AppState | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``app_state`` lets callers inject state (the module-level singleton is the
    default); each application instance owns its own state so repeated
    ``create_app()`` calls never share a live model.
    """
    shared_state = app_state if app_state is not None else AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure NLTK data is available before any predictions.
        ensure_stopwords_available()
        # Log configuration warnings before model load.
        for warning in settings.validate():
            logger.warning("Config: %s", warning)
        # Load the model/vectorizer once during startup; ModelService.load()
        # also fingerprints the artifact files and describes the model, so
        # readiness facts live with the service itself.
        logger.info(
            "Loading model from %s and vectorizer from %s",
            settings.model_file,
            settings.vectorizer_file,
        )
        service = ModelService(settings.model_file, settings.vectorizer_file)
        try:
            service.load()
        except ModelLoadError as exc:
            # Do not silently continue; the app is unusable for predictions.
            logger.error("Model load failed: %s", exc)
            shared_state.model = None
            raise RuntimeError(str(exc)) from exc
        shared_state.model = service
        logger.info(
            "Model loaded successfully (backend=%s model=%s vocab=%d "
            "model_sha256=%s vectorizer_sha256=%s)",
            service._backend,
            settings.model_file.name,
            service.vocab_size,
            service.model_sha256,
            service.vectorizer_sha256,
        )
        logger.info("Uncertainty threshold: %s", settings.UNCERTAINTY_THRESHOLD)
        logger.info("Top features: %s", settings.TOP_FEATURES)
        yield
        shared_state.model = None

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
    application.state.app_state = shared_state

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Bounds request bodies BEFORE any route reads them (audit B4): rejects
    # oversized payloads with 413 without ever materialising the body.
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.MAX_REQUEST_BODY_BYTES,
    )

    # Outermost: enforces the per-IP sliding-window rate limit (audit B5).
    application.add_middleware(RateLimitMiddleware)

    # Outermost-of-all: correlates every request (even rejected ones) with a
    # request id and emits a structured access log (audit B7).
    application.add_middleware(RequestIDMiddleware)

    @application.exception_handler(Exception)
    def unhandled_exception(request: Request, exc: Exception):
        """Return a clean, stack-trace-free 500 response for any unexpected error."""
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    @application.exception_handler(ScrapeError)
    def scrape_error(request: Request, exc: ScrapeError):
        """Map URL-analysis failures to a friendly 422 with a stable category.

        The technical ``detail`` and full redirect trace are only logged at
        DEBUG level and never exposed to the client.
        """
        logger.debug(
            "ScrapeError category=%s %s final_url=%s redirects=%s detail=%s",
            exc.category,
            exc,
            exc.final_url,
            list(exc.redirects),
            exc.detail,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "category": exc.category},
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
    def health(request: Request) -> HealthResponse:
        service = request.app.state.app_state.model
        model_loaded = bool(service and service.model_is_loaded)
        vectorizer_loaded = bool(service and service.vectorizer_is_loaded)
        if (service is not None and service.is_loaded
                and getattr(service, "_backend", None) is not None):
            model_file = service.model_file.name
            vectorizer_file = service.vectorizer_file.name
            model_backend = service._backend
            model_sha256 = getattr(service, "model_sha256", None)
            vectorizer_sha256 = getattr(service, "vectorizer_sha256", None)
            vocab_size = getattr(service, "vocab_size", None)
        else:
            model_file = vectorizer_file = model_backend = None
            model_sha256 = vectorizer_sha256 = vocab_size = None
        return HealthResponse(
            status="ok",
            model_loaded=model_loaded,
            vectorizer_loaded=vectorizer_loaded,
            model_backend=model_backend,
            model_file=model_file,
            vectorizer_file=vectorizer_file,
            model_sha256=model_sha256,
            vectorizer_sha256=vectorizer_sha256,
            vocab_size=vocab_size,
        )

    @application.get("/health/live", response_model=LiveResponse)
    def health_live() -> LiveResponse:
        """Liveness: the process serves HTTP regardless of model state.

        Deliberately model-independent so orchestrators can tell "up" apart
        from "ready" (audit B6 — the old blended /health conflated the two).
        """
        return LiveResponse(status="ok")

    @application.get("/health/ready", response_model=ReadyResponse)
    def health_ready(request: Request) -> ReadyResponse:
        """Readiness: can this instance actually produce a prediction?

        503 + detail while the model is still loading or failed to load;
        on success returns the same readiness facts the blended /health reports.
        """
        service = request.app.state.app_state.model
        model_loaded = bool(service and service.model_is_loaded)
        model_ready = bool(service and service.model_ready)
        if not model_ready:
            raise HTTPException(
                status_code=503,
                detail="Detector is not ready to serve predictions.",
            )
        return ReadyResponse(
            status="ready",
            model_loaded=model_loaded,
            model_ready=model_ready,
            model_backend=getattr(service, "_backend", None),
            model_file=service.model_file.name if service else None,
            vectorizer_file=service.vectorizer_file.name if service else None,
            model_sha256=getattr(service, "model_sha256", None),
            vectorizer_sha256=getattr(service, "vectorizer_sha256", None),
            vocab_size=getattr(service, "vocab_size", None),
        )

    @application.get("/info")
    def info():
        """Return application version and configuration summary."""
        return {
            "version": "2.0.0",
            "settings": settings.summary(),
        }

    @application.get("/history")
    def get_history(limit: int = 20):
        """Return recent predictions made via the API."""
        entries = prediction_log.recent(limit=limit)
        return [
            {
                "label": e.label,
                "confidence": e.confidence,
                "probability_real": e.probability_real,
                "probability_fake": e.probability_fake,
                "source_type": e.source_type,
                "input_preview": e.input_preview,
            }
            for e in entries
        ]

    def _require_model(request: Request) -> ModelService:
        model = request.app.state.app_state.model
        if model is None or not model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="The detector model is not loaded. Please try again later.",
            )
        return model

    def _to_response(
        service: ModelService,
        raw_text: str,
        source_type: str,
        source: str | None = None,
        page_title: str | None = None,
    ) -> PredictResponse:
        prediction = service.predict(raw_text)
        response = PredictResponse(
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
            source=source,
            page_title=page_title,
        )
        prediction_log.append(
            PredictionEntry(
                label=prediction.label,
                confidence=prediction.confidence,
                probability_real=prediction.probability_real,
                probability_fake=prediction.probability_fake,
                source_type=source_type,
                input_preview=raw_text[:100],
            )
        )
        return response

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: Request, req: PredictRequest) -> PredictResponse:
        service = _require_model(request)
        text = req.news.strip()
        if len(text) > settings.MAX_INPUT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Input text too long ({len(text)} chars). "
                       f"Maximum is {settings.MAX_INPUT_LENGTH} characters.",
            )
        return _to_response(service, text, "text")

    @application.post("/predict-url", response_model=PredictResponse)
    def predict_url(request: Request, req: UrlRequest) -> PredictResponse:
        service = _require_model(request)
        key = cache_key(req.url)
        cached = (
            shared_state.url_cache.get(key)
            if settings.CACHE_URL_ENABLED
            else None
        )
        if cached is not None:
            extract: ExtractResult = cached
        else:
            extract = fetch_article(req.url)
            if extract.text.strip() and settings.CACHE_URL_ENABLED:
                shared_state.url_cache.put(key, extract)
        if not extract.text.strip():
            raise HTTPException(
                status_code=422,
                detail="We retrieved the page, but couldn't identify the article content.",
            )
        return _to_response(
            service,
            extract.text,
            "url",
            source=req.url,
            page_title=extract.title or None,
        )

    return application


state = AppState()
app = create_app(app_state=state)
