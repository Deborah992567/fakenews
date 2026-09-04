"""Tests for the model verify_model module (structure only, no TF needed)."""

import pytest
from unittest import mock


def test_check_compatibility_passes_when_dims_match():
    from app.verify_model import _check_compatibility
    model = mock.Mock()
    model.input_shape = (None, 100)
    vectorizer = mock.Mock()
    vectorizer.vocabulary_ = {f"w{i}": i for i in range(100)}
    assert _check_compatibility(model, vectorizer) is True


def test_check_compatibility_fails_when_dims_differ():
    from app.verify_model import _check_compatibility
    model = mock.Mock()
    model.input_shape = (None, 100)
    vectorizer = mock.Mock()
    vectorizer.vocabulary_ = {f"w{i}": i for i in range(50)}
    assert _check_compatibility(model, vectorizer) is False


def test_main_returns_1_when_model_missing(tmp_path):
    from app.verify_model import main
    with mock.patch("app.verify_model.settings") as s:
        s.model_file = tmp_path / "nonexistent.h5"
        s.vectorizer_file = tmp_path / "nonexistent.pkl"
        result = main()
    assert result == 1
