# AGENTS.md — Developer Instructions for Fake News Detector

## Project Overview
A FastAPI-based fake news detection API with a trained Keras neural network,
explainability (gradient saliency), and a single-page frontend.

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
- `MODEL_PATH` (default `my_model.h5`)
- `VECTORIZER_PATH` (default `countvectorizer.pkl`)
- `UNCERTAINTY_THRESHOLD` (default 0.10)
- `MAX_INPUT_LENGTH` (default 20000)
- `LOG_LEVEL` (default INFO)

## Architecture
- `app/config.py` — Settings from env vars
- `app/main.py` — FastAPI app, lifespan, routes
- `app/model.py` — ModelService (TF/Keras model + CountVectorizer)
- `app/preprocessing.py` — NLTK text cleaning pipeline
- `app/schemas.py` — Pydantic request/response models
- `app/scraper.py` — URL fetching with SSRF protection
- `app/prediction_log.py` — Server-side prediction ring buffer
- `app/logging_config.py` — Logging setup
- `app/verify_model.py` — Standalone model verification script
- `frontend/` — Single-page HTML/CSS/JS frontend
- `tests/` — pytest test suite (100+ tests)

## Model Details
- Keras Sequential: Dense(12,relu) x3 → Dense(1, sigmoid)
- Input: CountVectorizer bag-of-words with NLTK Porter-stemmed tokens
- Label 1 = REAL, sigmoid output = P(real)
- probability_fake = 1 - probability_real
- Explainability: tf.GradientTape gradient-based saliency

## Code Conventions
- Python 3.12, type hints throughout
- Pydantic v2 for validation and serialization
- pytest for testing (no unittest)
- Frontend: vanilla JS, no frameworks
- Never merge to `main`; work only on `feature/detector-overhaul`
