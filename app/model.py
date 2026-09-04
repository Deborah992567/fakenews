"""Model wrapper around the trained fake-news detector.

Loads the TensorFlow/Keras model and the scikit-learn vectorizer once at
application startup, then exposes a single ``predict`` entry point used by
both the pasted-text and URL endpoints.

Input attribution (explainability) is computed using gradient-based saliency:
we measure the gradient of the model's real-probability output with respect to
each active (present) word in the input. A positive attribution means the word
pushes the prediction toward *real*; a negative attribution pushes toward
*fake*. These are *model influences*, not factual proof that a word is real or
fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app import preprocessing
from app.config import settings


class ModelLoadError(Exception):
    """Raised when the model or vectorizer cannot be loaded."""


@dataclass
class ExplanationItem:
    """A single word's contribution to a prediction."""

    word: str
    impact: float
    direction: str


@dataclass
class Prediction:
    """A complete prediction result."""

    probability_real: float
    probability_fake: float
    label: str
    confidence: float
    explanation: list[ExplanationItem] = field(default_factory=list)


class ModelService:
    """Encapsulates the trained model, vectorizer and prediction logic."""

    def __init__(self, model_file: Path, vectorizer_file: Path) -> None:
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self._model: Any | None = None
        self._vectorizer: Any | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> "ModelService":
        """Load the model and vectorizer from disk.

        Raises :class:`ModelLoadError` with a clear message identifying the
        missing or unloadable file.
        """
        self._model = self._load_model()
        self._vectorizer = self._load_vectorizer()

        if self._model is None or self._vectorizer is None:
            raise ModelLoadError("Model or vectorizer failed to initialise.")
        return self

    def _load_model(self) -> Any | None:
        if not self.model_file.exists():
            raise ModelLoadError(
                f"Model file not found at {self.model_file}. "
                "Place the trained Keras model there (or set MODEL_PATH)."
            )
        import tensorflow as tf

        try:
            loaded = tf.keras.models.load_model(self.model_file, compile=False)
        except Exception as exc:  # noqa: BLE001 - surface any keras load failure
            raise ModelLoadError(
                f"Failed to load the model from {self.model_file}: {exc}"
            ) from exc
        return loaded

    def _load_vectorizer(self) -> Any | None:
        if not self.vectorizer_file.exists():
            raise ModelLoadError(
                f"Vectorizer file not found at {self.vectorizer_file}. "
                "Place the trained CountVectorizer there (or set VECTORIZER_PATH)."
            )
        try:
            import pickle

            with self.vectorizer_file.open("rb") as handle:
                return pickle.load(handle)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load the vectorizer from {self.vectorizer_file}: {exc}"
            ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None

    @property
    def model_is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vectorizer_is_loaded(self) -> bool:
        return self._vectorizer is not None

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _probability_real(self, cleaned_text: str) -> float:
        """Return the raw probability that the text is real."""
        vector = self._vectorizer.transform([cleaned_text]).toarray()
        raw = float(self._model.predict(vector, verbose=0)[0][0])
        # Guard against a single scalar output being returned as-is.
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw = float(raw[0])
        return float(np.clip(raw, 0.0, 1.0))

    def predict(self, raw_text: str) -> Prediction:
        """Run the full pipeline for raw text and return a Prediction."""
        cleaned = preprocessing.clean_single_text(raw_text)
        prob_real = self._probability_real(cleaned)
        prob_fake = 1.0 - prob_real
        label, confidence = self._verdict(prob_real, prob_fake)
        explanation = self._explain(cleaned)
        return Prediction(
            probability_real=prob_real,
            probability_fake=prob_fake,
            label=label,
            confidence=confidence,
            explanation=explanation,
        )

    def _verdict(self, prob_real: float, prob_fake: float) -> tuple[str, float]:
        """Return (label, confidence) based on the uncertainty threshold."""
        winner = max(prob_real, prob_fake)
        if winner - 0.5 < settings.UNCERTAINTY_THRESHOLD:
            return "uncertain", round(winner * 100.0, 2)
        label = "real" if prob_real >= prob_fake else "fake"
        return label, round(winner * 100.0, 2)

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def _explain(self, cleaned_text: str) -> list[ExplanationItem]:
        """Compute per-word influence via gradient attribution.

        Returns the strongest contributing words with a direction of "real"
        or "fake" based on the sign of the gradient against the vectorizer's
        feature index for each word present in the input.
        """
        vector = self._vectorizer.transform([cleaned_text]).toarray()
        active_indices = np.flatnonzero(vector[0])
        if len(active_indices) == 0:
            return []

        try:
            import tensorflow as tf

            feature_names = self._feature_names()
            input_tensor = tf.convert_to_tensor(vector.astype("float32"))
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                output = self._model(input_tensor, training=False)
            gradients = tape.gradient(output, input_tensor).numpy()[0]
        except Exception:  # noqa: BLE001 - fall back to no explanation
            return []

        # For each active word, identify the cleaned (stemmed) token.
        words = cleaned_text.split()
        word_to_index: dict[str, int] = {}
        for idx in active_indices:
            for word in words:
                feature = feature_names[idx]
                if feature == word:
                    word_to_index.setdefault(word, idx)
                    break

        contributions: list[ExplanationItem] = []
        for idx in active_indices:
            word = feature_names[idx]
            grad = float(gradients[idx])
            if abs(grad) < 1e-9:
                continue
            direction = "real" if grad > 0 else "fake"
            contributions.append(
                ExplanationItem(
                    word=word,
                    impact=float(round(abs(grad) * 100.0, 2)),
                    direction=direction,
                )
            )

        contributions.sort(key=lambda item: item.impact, reverse=True)
        return contributions[: settings.TOP_FEATURES]

    def _feature_names(self) -> list[str]:
        """Return the vectorizer's feature names (works across sklearn versions)."""
        if hasattr(self._vectorizer, "get_feature_names_out"):
            return list(self._vectorizer.get_feature_names_out())
        return list(self._vectorizer.get_feature_names())
