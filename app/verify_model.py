"""Model startup verification.

Can be run standalone to verify that the model and vectorizer files
are loadable and compatible, independent of the FastAPI server:

    python -m app.verify_model
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from app.config import settings
from app.preprocessing import clean_single_text


def _load_model(path: Path):
    import tensorflow as tf

    print(f"Loading model from {path} ...")
    model = tf.keras.models.load_model(path, compile=False)
    print(f"  Model type : {type(model).__name__}")
    print(f"  Parameters : {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    return model


def _load_vectorizer(path: Path):
    import pickle

    print(f"Loading vectorizer from {path} ...")
    with path.open("rb") as f:
        cv = pickle.load(f)
    vocab_size = len(cv.vocabulary_)
    print(f"  Type       : {type(cv).__name__}")
    print(f"  Vocab size : {vocab_size}")
    return cv


def _check_compatibility(model, vectorizer):
    input_dim = model.input_shape[-1]
    vocab_size = len(vectorizer.vocabulary_)
    print(f"\nCompatibility check:")
    print(f"  Model input dim : {input_dim}")
    print(f"  Vectorizer vocab: {vocab_size}")
    if input_dim == vocab_size:
        print("  Status: COMPATIBLE")
        return True
    else:
        print("  Status: MISMATCH — model input dim != vectorizer vocab size")
        return False


def _test_prediction(model, vectorizer):
    print(f"\nRunning test prediction ...")
    sample = clean_single_text(
        "Breaking news today as government announces new economic policy"
    )
    print(f"  Cleaned text: {sample[:80]}...")
    vector = vectorizer.transform([sample]).toarray()
    print(f"  Vector shape: {vector.shape}")
    print(f"  Non-zero features: {np.count_nonzero(vector)}")
    prediction = model.predict(vector, verbose=0)
    prob_real = float(prediction[0][0])
    prob_fake = 1.0 - prob_real
    label = "real" if prob_real >= prob_fake else "fake"
    print(f"  P(real)     : {prob_real:.4f}")
    print(f"  P(fake)     : {prob_fake:.4f}")
    print(f"  Label       : {label}")
    print(f"\n  Gradient-based explainability test ...")
    import tensorflow as tf

    input_tensor = tf.convert_to_tensor(vector.astype("float32"))
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor, training=False)
    gradients = tape.gradient(output, input_tensor)
    if gradients is not None:
        grad_np = gradients.numpy()[0]
        n_active = np.count_nonzero(grad_np)
        print(f"  Gradient shape  : {grad_np.shape}")
        print(f"  Non-zero grads  : {n_active}")
        print(f"  Gradient valid  : YES")
    else:
        print(f"  Gradient valid  : NO (gradients are None)")
    return True


def main() -> int:
    model_path = settings.model_file
    vectorizer_path = settings.vectorizer_file

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    if not vectorizer_path.exists():
        print(f"ERROR: Vectorizer not found at {vectorizer_path}")
        return 1

    try:
        model = _load_model(model_path)
        vectorizer = _load_vectorizer(vectorizer_path)
        compatible = _check_compatibility(model, vectorizer)
        if not compatible:
            return 1
        _test_prediction(model, vectorizer)
        print(f"\nAll checks passed.")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
