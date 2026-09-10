FROM python:3.12-slim

# Prevent bytecode noise, buffered output and pip cache bloat.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

# Copy application source and static frontend.
COPY main.py .
COPY app/ ./app/
COPY frontend/ ./frontend/

# Ship ONLY the promoted production artifacts (TF-IDF LogisticRegression).
# The legacy Keras model (artifacts/baseline/*) is deliberately NOT copied.
COPY my_model_lr.pkl my_tfidf_vectorizer.pkl ./

# Run as an unprivileged user. The detector only ever READS the model files,
# so the artifacts are made read-only; NLTK data (and any runtime scratch) is
# owned by the app user under /app/nltk_data.
RUN addgroup --system appuser \
    && adduser --system --ingroup appuser --home /app appuser \
    && mkdir -p /app/nltk_data \
    && chown -R appuser:appuser /app \
    && chmod 444 /app/my_model_lr.pkl /app/my_tfidf_vectorizer.pkl

USER appuser

EXPOSE 8000

# Gate on readiness so the container is only "healthy" once the promoted
# detector is loaded AND described (see app/model.py model_ready).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

CMD ["python", "main.py"]