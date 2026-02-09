"""
Stage 3 — Per-Rep Assessment.

Segments individual repetitions from the raw landmark sequence using
exercise/view-specific biomechanical signals, then runs each rep through
a pre-trained temporal CNN to produce 5 aspect-level quality scores (0-10).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import (
    ASSESSMENT_MODELS_BASE_DIR,
    EXERCISE_MAP,
    FEATURE_DIM,
    PROJECT_ROOT,
    T_FIXED,
)

# ---------------------------------------------------------------------------
# Direct imports — bypass __init__.py chains (avoids seaborn dependency).
# ---------------------------------------------------------------------------
import importlib.util as _ilu


def _import_module_from_file(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_qa = _import_module_from_file(
    "quality_assessment",
    str(PROJECT_ROOT / "src" / "scripts" / "quality_assessment.py"),
)
_preprocess = _import_module_from_file(
    "preprocess_pose_RGB",
    str(PROJECT_ROOT / "src" / "preprocessing" / "preprocess_pose_RGB.py"),
)

segment_reps_from_sequence = _qa.segment_reps_from_sequence
_load_model_checkpoint = _qa._load_model_checkpoint
_attention_pool_reps = _qa._attention_pool_reps
validate_feature_dimensions = _qa.validate_feature_dimensions
safe_name = _qa.safe_name
_resample_sequence = _preprocess._resample_sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache: (exercise_name, view) → (model, meta)
# ---------------------------------------------------------------------------
_assessment_cache: dict[tuple[str, str], tuple[tf.keras.Model, dict]] = {}


def _models_dir_for_view(view: str) -> str:
    """Return the assessment models directory for a given view."""
    return str(ASSESSMENT_MODELS_BASE_DIR / f"{view}_37feat")


def load_assessment_model(
    exercise_name: str,
    view: str,
) -> Tuple[tf.keras.Model, dict]:
    """Load (or return cached) per-exercise assessment model + metadata.

    Models are stored as ``<safe_name>_best.keras`` + ``<safe_name>_meta.json``
    inside ``src/models/assessment_models/{view}_37feat/``.

    Args:
        exercise_name: Human-readable exercise name (e.g. "Dumbbell shoulder press").
        view: ``"front"`` or ``"side"``.

    Returns:
        Tuple of (Keras model, metadata dict with ``aspect_cols``, etc.).

    Raises:
        FileNotFoundError: If the model or meta file is missing.
    """
    key = (exercise_name, view)
    if key in _assessment_cache:
        return _assessment_cache[key]

    models_dir = _models_dir_for_view(view)
    logger.info("Loading assessment model for '%s' (%s) from %s", exercise_name, view, models_dir)

    model, meta = _load_model_checkpoint(models_dir, exercise_name)
    _assessment_cache[key] = (model, meta)
    return model, meta


def assess_reps(
    lm_seq_xyz: np.ndarray,
    exercise_name: str,
    exercise_id: int,
    view: str,
) -> Tuple[List[dict], dict]:
    """Segment reps and run per-rep assessment.

    Args:
        lm_seq_xyz: Shape (N, 33, 3) — raw (un-normalized) landmarks from Stage 1.
            ``segment_reps_from_sequence`` normalizes internally.
        exercise_name: Exercise name (must match training data naming).
        exercise_id: 1-indexed exercise ID.
        view: ``"front"`` or ``"side"``.

    Returns:
        Tuple of:
          - per_rep_scores: List of dicts, each containing
            ``{"rep_number": int, "scores": {aspect_name: float (0-10)}}``.
          - debug_info: Dict with segmentation diagnostics.

    Raises:
        FileNotFoundError: If the assessment model is missing.
        RuntimeError: If no reps are detected.
    """
    # --- Load model & meta ---
    model, meta = load_assessment_model(exercise_name, view)
    aspect_cols: list[str] = meta["aspect_cols"]
    n_aspects = meta["n_aspects"]
    ckpt_feat_dim = int(meta.get("feature_dim", FEATURE_DIM))

    if ckpt_feat_dim != FEATURE_DIM:
        raise RuntimeError(
            f"Feature dimension mismatch for '{exercise_name}': "
            f"checkpoint expects {ckpt_feat_dim}, pipeline provides {FEATURE_DIM}."
        )

    # --- Segment reps ---
    reps, peaks, signal, used = segment_reps_from_sequence(
        exercise=exercise_name,
        view=view,
        lm_seq_xyz=lm_seq_xyz,
        feature_mode="37",
    )

    n_reps = int(reps.shape[0])
    logger.info(
        "Segmentation for '%s': %d reps detected (peaks=%s)",
        exercise_name, n_reps, peaks,
    )

    if n_reps == 0:
        raise RuntimeError(
            "No valid repetitions were detected. "
            "Please try again with a clearer view."
        )

    validate_feature_dimensions(reps, expected_dim=FEATURE_DIM)

    # --- Per-rep CNN inference ---
    # reps shape: (R, 2*win, 37).  The CNN expects (B, T_fixed, 37),
    # so we resample each rep window to T_fixed if needed.
    rep_inputs = []
    for i in range(n_reps):
        rep_window = reps[i]  # (2*win, 37)
        if rep_window.shape[0] != T_FIXED:
            rep_window = _resample_sequence(rep_window, T_FIXED)
        rep_inputs.append(rep_window)

    rep_batch = np.array(rep_inputs, dtype=np.float32)  # (R, T_FIXED, 37)
    raw_preds = model.predict(rep_batch, verbose=0)      # (R, n_aspects) in [0, 1]
    scores_0_10 = raw_preds * 10.0                        # scale to 0-10

    # --- Build per-rep score dicts ---
    per_rep_scores: list[dict] = []
    for i in range(n_reps):
        rep_dict = {
            "rep_number": i + 1,
            "scores": {
                aspect_cols[j]: round(float(scores_0_10[i, j]), 2)
                for j in range(n_aspects)
            },
        }
        per_rep_scores.append(rep_dict)

    debug_info = {
        "peaks": [int(p) for p in peaks],
        "n_reps": n_reps,
        "used_params": used,
        "signal_stats": {
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "mean": float(np.mean(signal)),
        },
    }

    return per_rep_scores, debug_info


def aggregate_scores(per_rep_scores: list[dict]) -> Tuple[dict[str, float], float]:
    """Compute mean per-aspect scores and overall score across reps.

    Args:
        per_rep_scores: Output of ``assess_reps``.

    Returns:
        Tuple of (aspect_means: {name: float}, overall_mean: float).
    """
    if not per_rep_scores:
        return {}, 0.0

    aspect_accum: dict[str, list[float]] = {}
    for rep in per_rep_scores:
        for aspect, score in rep["scores"].items():
            aspect_accum.setdefault(aspect, []).append(score)

    aspect_means = {
        aspect: round(float(np.mean(vals)), 2)
        for aspect, vals in aspect_accum.items()
    }
    overall = round(float(np.mean(list(aspect_means.values()))), 2)
    return aspect_means, overall
