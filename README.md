# Fake News Detector

A web application that analyses whether a piece of news text is likely to be
**real** or **fake**, using a pre-trained machine learning model. It supports
both pasted text and article URLs, returns an actionable verdict with
confidence and probability scores, and explains the prediction by surfacing
the words that most influenced the model.

Built with **FastAPI** on the backend and a dependency-free **vanilla
HTML/CSS/JS** frontend.

---

## Features

- **Paste-text analysis** – analyse raw article text.
- **URL analysis** – fetch a page, extract its article content and analyse it
  through the same pipeline (with SSRF protection, timeouts and size limits).
- **Verdict with confidence** – `real`, `fake` or `uncertain`.
- **Uncertainty handling** – configurable threshold; near 50/50 predictions are
  reported as `uncertain` rather than forced into a confident answer.
- **Explainability** – returns the top words that influenced the prediction and
  whether they pushed toward *real* or *fake*.
- **Prediction history** – stored client-side using `localStorage`.
- **Frontend served by FastAPI** – no separate development server.
- **Docker support** – run the whole application with a single command.
- **API documentation** – automatic OpenAPI docs at `/docs` and `/redoc`.

---

## Architecture overview

```
Pasted text                      URL
     │                            │
     ▼                            ▼
preprocessing               fetch + article extraction
     │                            │
     └──────────►    vectorizer    ◄──────────┘
                            │
                            ▼
                       model (Keras)
                            │
                            ▼
                      prediction
                            │
                            ▼
                      explainability
```

Both the pasted-text and URL endpoints share a single prediction pipeline
(`app/model.py`), so behaviour is consistent regardless of input source.

```
app/
├── __init__.py
├── main.py          # FastAPI app, lifespan model loading, routes
├── config.py        # environment-based configuration
├── model.py         # model/vectorizer loading + shared prediction pipeline
├── preprocessing.py # text cleaning: regex, lowercase, stopwords, stemming
├── explainability   # gradient-based word attribution (in model.py)
├── scraper.py       # safe URL fetching + article extraction
└── schemas.py       # Pydantic request/response models
frontend/
├── index.html
├── style.css
└── script.js
```

---

## Requirements

- Python **3.10 – 3.12** (TensorFlow compatibility)
- pip

> TensorFlow does not yet support Python 3.13+, so use a 3.12 environment.

---

## Installation

### 1. Create a virtual environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
```

Windows (Command Prompt):

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Test dependencies (optional):

```bash
pip install -r requirements-dev.txt
```

### 3. Model / vectorizer files

The application expects two trained artifacts in the project root (or at paths
configured via `MODEL_PATH` and `VECTORIZER_PATH`):

- `my_model.h5` – the trained Keras neural network.
- `countvectorizer.pkl` – a scikit-learn `CountVectorizer` fitted on the
  training corpus.

These reproduce the pipeline defined in `fake_news.ipynb`. If a file is
missing, the application fails to start with a clear message rather than
silently serving broken predictions.

---

## Configuration

Configuration is read from environment variables (optionally via a `.env`
file). See `.env.example` for a full template:

| Variable                 | Default         | Description                                        |
| ------------------------ | --------------- | -------------------------------------------------- |
| `HOST`                 | `0.0.0.0`       | Bind host for the server.                          |
| `PORT`                 | `8000`          | Port for the server.                               |
| `LOG_LEVEL`            | `INFO`          | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `MODEL_PATH`             | `my_model.h5`   | Path to the trained model.                         |
| `VECTORIZER_PATH`        | `countvectorizer.pkl` | Path to the vectorizer.                    |
| `UNCERTAINTY_THRESHOLD`  | `0.10`          | Distance from 50% below which a verdict is uncertain. |
| `MAX_INPUT_LENGTH`       | `20000`         | Maximum characters accepted for pasted text.       |
| `MAX_URL_RESPONSE_SIZE`  | `1000000`       | Maximum bytes accepted from a fetched URL.         |
| `REQUEST_TIMEOUT`        | `10`            | URL fetch timeout in seconds.                      |
| `MAX_REDIRECTS`          | `5`             | Maximum redirects while fetching a URL.            |
| `CORS_ORIGINS`           | `*`             | Comma-separated allowed origins.                   |

Copy the template and adjust as needed:

```bash
cp .env.example .env
```

---

## Running locally

```bash
python main.py
```

Or using the Makefile:

```bash
make run
```

Then open <http://localhost:8000/> in a browser.

---

## Docker

Build and start with a single command:

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

Then open <http://localhost:8000/>.

---

## API endpoints

| Method | Path          | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| GET    | `/`           | Serves the frontend.                         |
| GET    | `/health`     | Health/readiness check.                      |
| POST   | `/predict`    | Analyse pasted article text.                 |
| POST   | `/predict-url`| Analyse an article at a URL.                 |
| GET    | `/docs`       | Interactive API documentation (Swagger UI).  |
| GET    | `/redoc`      | Alternative API documentation.               |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### `POST /predict` — example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"news": "A large earthquake struck the region early this morning..."}'
```

### Example response

```json
{
  "label": "real",
  "confidence": 87.42,
  "probability_real": 87.42,
  "probability_fake": 12.58,
  "explanation": {
    "top_influential_words": [
      { "word": "govern", "impact": 1.24, "direction": "real" },
      { "word": "report", "impact": 0.96, "direction": "real" }
    ]
  },
  "source_type": "text"
}
```

Values are real JSON numbers. `label` is one of `real`, `fake`, `uncertain`.

### Validation

Missing, empty, whitespace-only or near-empty `news` returns HTTP `422`:

```json
{
  "detail": "News text must contain enough content to analyze."
}
```

---

## URL analysis

`POST /predict-url` accepts an article URL:

```json
{
  "url": "https://example.com/article"
}
```

The backend validates the URL (HTTP/HTTPS only, blocks private/localhost
networks to prevent SSRF), fetches the page with a timeout and size limit,
extracts the main article text, and runs it through the same prediction
pipeline. It returns the same schema as `/predict`.

```bash
curl -X POST http://localhost:8000/predict-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

---

## Explainability

The `explanation.top_influential_words` array lists the words that most
influenced the prediction together with their `impact` and `direction`
(`real` or `fake`). These are **model features/influences**, computed via
gradient attribution — they are **not proof** that a word is real or fake.

---

## Uncertainty handling

The winning probability must exceed 50% by more than `UNCERTAINTY_THRESHOLD`
(in probability, default `0.10`). Otherwise the label is `uncertain`. The raw
probabilities are still returned so callers always see the underlying model
scores.

---

## Testing

```bash
pytest
```

The test suite covers preprocessing, `/predict` validation and schema,
uncertainty handling, `/health`, and URL validation (with external HTTP
mocked — no real network or news site is contacted).

---

## Project structure

```
.
├── app/                  # FastAPI backend package
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── scraper.py
│   └── schemas.py
├── frontend/             # static frontend (no build step)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/                # pytest suite
├── my_model.h5           # trained model (tracked)
├── countvectorizer.pkl   # trained vectorizer (tracked)
├── fake_news.ipynb       # original training notebook
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tensorflow'`** – make sure you are
  using Python 3.10–3.12 and ran `pip install -r requirements.txt`.
- **Model fails to load at startup** – verify `my_model.h5` exists and matches
  `MODEL_PATH`, and that the vectorizer was fitted with the same preprocessing
  as in `fake_news.ipynb`.
- **Port already in use** – set a different `PORT` in your environment or
  `.env`.
- **Frontend shows "Unable to connect"** – the backend is not running; start it
  with `python main.py`.
- **Docker build is slow** – the TensorFlow wheel is large (~250 MB); this is
  expected on the first build.

---

## Known limitations

- **TensorFlow is a large dependency (~250 MB wheel)**.  On slow or
  metered networks the initial `pip install -r requirements.txt` may
  take several minutes or time out.  Retry with a longer timeout
  (`pip install --timeout 600 …`) or install behind a fast network.
- **The existing model is a Keras Sequential neural network** with four
  Dense layers and a sigmoid output.  It was trained on a balanced
  real/fake news dataset and expects bag-of-words CountVectorizer input
  with NLTK Porter-stemmed tokens.  The vectorizer and model are
  tightly coupled; replacing one without the other will produce
  incorrect results.
- **Explainability is gradient-based**.  Because the model is a neural
  network (not a linear classifier), feature-importance values are
  computed via `tf.GradientTape` rather than model coefficients.  These
  are *model influences*, not factual proof that a word makes an
  article real or fake.
- **Prediction history is client-side only** (browser `localStorage`).
  It is not shared across devices or browsers.
- **URL analysis depends on external sites** being reachable and
  returning parseable HTML.  Some sites block scrapers; results
  for those URLs will return an error message.
- **Uncertainty handling** uses a configurable threshold
  (`UNCERTAINTY_THRESHOLD`).  The default value (0.10) may not be
  appropriate for all use-cases.  Tune it for your domain.
