# AGENTS.md — Developer Instructions for Fake News Detector

## Project Overview
A FastAPI-based fake news detection API with a promoted scikit-learn
TF-IDF LogisticRegression detector (explainability via influential-feature
contributions), and a single-page frontend. A legacy Keras network is preserved
as a rollback backup.

## Build & Run
```bash
# Install dependencies (requires Python 3.10-3.12)
pip install -r requirements.txt

# Start dev server
python main.py
# or
uvicorn app.main:app --reload

# Open browser at http://localhost:8000
```

## Testing
```bash
# Run all tests (requires pytest + httpx)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_predict.py -v
```

## Key Commands
- `make install` — install runtime dependencies
- `make install-dev` — install runtime + test dependencies
- `make test` — run pytest
- `make fmt` — compile-check all Python files
- `make docker-up` — build and start with Docker Compose
- `make docker-down` — stop Docker Compose

## Environment Variables
All configurable via env or `.env` file (see `.env.example`):
- `PORT` (default 8000)
- `MODEL_PATH` (default `my_model_lr.pkl` — promoted TF-IDF LogisticRegression)
- `VECTORIZER_PATH` (default `my_tfidf_vectorizer.pkl`)
- `UNCERTAINTY_THRESHOLD` (default 0.10)
- `MAX_INPUT_LENGTH` (default 20000)
- `MAX_URL_LENGTH` (default 2048 — centralized URL cap)
- `MAX_REQUEST_BODY_BYTES` (default 300000 — 413 beyond this)
- `CACHE_URL_ENABLED` / `CACHE_URL_TTL_SECONDS` / `CACHE_URL_MAX_ITEMS` — bounded
  in-memory cache of URL extraction results (defaults true/600/512)
- `RATE_LIMIT_ENABLED` / `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` /
  `RATE_LIMIT_MAX_IPS` — per-IP sliding-window rate limit (default true/120/60/10000)
- `LOG_LEVEL` (default INFO)

Legacy Keras artifacts `my_model.h5`/`countvectorizer.pkl` are preserved as
rollback backups under `artifacts/baseline/` (hashes in
`reports/release_manifest.json` and `reports/baseline_sha256.txt`).

## Architecture
- `app/config.py` — Settings from env vars
- `app/main.py` — FastAPI app, lifespan, routes, middleware wiring
- `app/model.py` — ModelService (sklearn LR/TF-IDF or legacy Keras + vectorizer)
- `app/preprocessing.py` — NLTK text cleaning pipeline
- `app/schemas.py` — Pydantic request/response models
- `app/scraper.py` — URL fetching with SSRF protection + thread-local sessions
- `app/artifacts.py` — artifact fingerprinting (SHA-256)
- `app/cache.py` — bounded TTL cache (URL extraction memoization)
- `app/ratelimit.py` — per-IP sliding-window rate limiter + middleware
- `app/security.py` — request-body size limiting middleware
- `app/observability.py` — request-id correlation + structured access logging
- `app/prediction_log.py` — Server-side prediction ring buffer
- `app/logging_config.py` — Logging setup (request-id aware)
- `app/verify_model.py` — Standalone model verification script
- `frontend/` — Single-page HTML/CSS/JS frontend
- `tests/` — pytest test suite (270+ tests)

## Model Details
- Promoted production model: scikit-learn LogisticRegression on TF-IDF features
  (sublinear TF, min_df=2, 36,862 terms), fitted ONLY on the training split of a
  cleaned ISOT + BuzzFeed-v02 corpus. P(real) = predict_proba[:,1]; verdict uses
  the 0.10 uncertainty band. Explainability uses linear feature contribution
  (coefficient × TF-IDF weight), a model-appropriate attribution, labelled
  "influential features" (not proof of truth/falsity).
- Legacy Keras model (kept as rollback backup under `artifacts/baseline/`):
  Dense(12,relu)^3 → sigmoid on CountVectorizer BOW; label 1 = REAL;
  P(fake) = 1 − P(real). Its explainability is tf.GradientTape saliency.

## Code Conventions
- Python 3.12, type hints throughout
- Pydantic v2 for validation and serialization
- pytest for testing (no unittest)
- Frontend: vanilla JS, no frameworks
- Never merge to `main`; work only on `feature/detector-overhaul`

## Phase 10 Notes (scalability & hardening)
- Run the full suite with `pytest` (272 tests). Do not weaken or delete tests.
- Model artifacts are FROZEN; every load fingerprints them and tests pin the
  exact SHA-256 digests from `reports/release_manifest.json`.
- Rate limiter and URL cache are per-process, in-memory, bounded and fail-open;
  see `docs/concurrency.md` for the documented multi-worker limits.
- New middleware (rate limit, body-size, request-id) is registered in
  `create_app()`; keep them outermost/ordered as documented in main.py.
- Middleware behavior is covered by `tests/test_ratelimit.py`,
  `tests/test_security.py`, `tests/test_observability.py`; cache by
  `tests/test_cache.py`; lifecycle by `tests/test_model_lifecycle.py` and
  `tests/test_app_lifecycle.py`.
