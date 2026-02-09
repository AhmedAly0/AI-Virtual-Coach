"""
Stage 2 — Exercise Recognition.

Loads a pre-trained 15-class MLP (.keras), flattens the (50, 37) feature
tensor to 1850-d, and returns the predicted exercise name, 1-indexed ID,
and softmax confidence.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import (
    RECOGNITION_MODELS_DIR,
    INT_TO_LABEL,
    EXERCISE_NAME_TO_ID,
    NUM_CLASSES,
    T_FIXED,
    FEATURE_DIM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache (populated by ``load_recognition_model``)
# ---------------------------------------------------------------------------
_recognition_models: dict[str, tf.keras.Model] = {}


def _model_path_for_view(view: str) -> Path:
    """Return the .keras path for the given camera view.

    Actual on-disk names are ``front_view.keras`` / ``side_view.keras``
    inside the (typo) ``exercise_reognition_models/`` directory.
    """
    return RECOGNITION_MODELS_DIR / f"{view}_view.keras"


def load_recognition_model(view: str) -> tf.keras.Model:
    """Load (or return cached) recognition model for *view*.

    Args:
        view: ``"front"`` or ``"side"``.

    Returns:
        Compiled Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if view in _recognition_models:
        return _recognition_models[view]

    path = _model_path_for_view(view)
    if not path.exists():
        raise FileNotFoundError(
            f"Recognition model not found at {path}. "
            f"Train with exercise_recognition.py first."
        )

    logger.info("Loading recognition model: %s", path)
    model = tf.keras.models.load_model(str(path), compile=False)
    _recognition_models[view] = model
    return model


def recognize_exercise(
    features_50: np.ndarray,
    view: str,
) -> Tuple[int, str, float]:
    """Run exercise recognition on a (50, 37) feature tensor.

    Args:
        features_50: Shape (T_FIXED, FEATURE_DIM).
        view: ``"front"`` or ``"side"``.

    Returns:
        Tuple of (exercise_id_1indexed, exercise_name, confidence).

    Raises:
        FileNotFoundError: If the model is missing.
        RuntimeError: On unexpected prediction shapes.
    """
    model = load_recognition_model(view)

    # Flatten (50, 37) → (1850,) and add batch dimension
    flat = features_50.reshape(1, -1).astype(np.float32)
    expected_dim = T_FIXED * FEATURE_DIM
    if flat.shape[1] != expected_dim:
        raise RuntimeError(
            f"Flattened input dim {flat.shape[1]} != expected {expected_dim}."
        )

    probs = model.predict(flat, verbose=0)  # (1, NUM_CLASSES)

    if probs.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Model output has {probs.shape[1]} classes, expected {NUM_CLASSES}."
        )

    class_idx = int(np.argmax(probs[0]))          # 0-indexed
    confidence = float(np.max(probs[0]))
    exercise_name = INT_TO_LABEL[class_idx]
    exercise_id = EXERCISE_NAME_TO_ID.get(exercise_name, -1)

    logger.info(
        "Recognition: '%s' (id=%d, class_idx=%d) confidence=%.3f",
        exercise_name, exercise_id, class_idx, confidence,
    )
    return exercise_id, exercise_name, confidence
