"""Real-model integration tests.

These tests exercise the actual trained TensorFlow/Keras model and the real
CountVectorizer.  They are skipped automatically when TensorFlow or the model
assets are unavailable, so the rest of the suite still runs in lightweight
environments.
"""

import numpy as np
import pytest

from app.config import settings
from app.model import ModelService, _has_input_gradients, _rebuild_functional_model


def _assets_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return settings.model_file.exists() and settings.vectorizer_file.exists()


pytestmark = pytest.mark.skipif(
    not _assets_available(),
    reason="TensorFlow or model/vectorizer assets are not available",
)


def _load_service() -> ModelService:
    return ModelService(settings.model_file, settings.vectorizer_file).load()


class TestRealLoading:
    def test_model_and_vectorizer_load(self):
        service = _load_service()
        assert service.is_loaded
        assert service.model_is_loaded
        assert service.vectorizer_is_loaded

    def test_input_dimension_matches_vocab(self):
        service = _load_service()
        input_dim = service._model.input_shape[-1]
        vocab_size = len(service._vectorizer.vocabulary_)
        assert input_dim == vocab_size

    def test_model_output_is_sigmoid_style(self):
        service = _load_service()
        vector = service._vectorizer.transform(["authorit"]).toarray()
        prob_real = float(service._model.predict(vector, verbose=0)[0][0])
        assert 0.0 <= prob_real <= 1.0


class TestRealPrediction:
    def test_probabilities_sum_to_one(self):
        svc = _load_service()
        pred = svc.predict(
            "Government announces new economic policy with broad support"
        )
        assert 0.0 <= pred.probability_real <= 1.0
        assert 0.0 <= pred.probability_fake <= 1.0
        assert abs((pred.probability_real + pred.probability_fake) - 1.0) < 1e-6
        assert isinstance(pred.confidence, float)

    def test_label_is_verdict(self):
        svc = _load_service()
        for text in [
            "Breaking news on the latest election results in the capital",
            "A completely normal article about weather and sports today",
            "Experts announce a major new discovery in medical research",
        ]:
            pred = svc.predict(text)
            assert pred.label in ("real", "fake", "uncertain")

    def test_explainability_returns_structured_words(self):
        svc = _load_service()
        pred = svc.predict(
            "Scientists announce the new mission control center opened downtown"
        )
        if pred.explanation:
            for item in pred.explanation:
                assert item.word
                assert isinstance(item.impact, float)
                assert item.direction in ("real", "fake")


class TestRebuildGradients:
    def test_model_has_input_gradients_after_rebuild(self):
        service = _load_service()
        assert _has_input_gradients(service._model)

    def test_rebuild_preserves_weights(self):
        service = _load_service()
        rebuilt = _rebuild_functional_model(service._model)
        original_weights = service._model.get_weights()
        for original, rebuilt_weight in zip(original_weights, rebuilt.get_weights()):
            assert np.allclose(original, rebuilt_weight)