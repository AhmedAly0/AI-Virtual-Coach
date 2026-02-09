"""
Stage 1 — Pose Preprocessing & Feature Engineering.

Receives raw pose landmarks from the mobile app (frames × 33 × 4) and produces
a (N, 37) feature matrix ready for recognition and assessment.

Pipeline:
    1. Validate input shape (N, 33, 4)
    2. Filter low-visibility frames
    3. Drop visibility column → (N, 33, 3)
    4. Normalize landmarks (pelvis-centered, torso-length-scaled)
    5. Compute 37 features per frame (19 base + 18 view-specialized)
    6. Resample to T=50 frames → (50, 37)
"""

import logging
from typing import Tuple

import numpy as np

from .config import (
    FEATURE_DIM,
    MIN_FRAMES,
    PROJECT_ROOT,
    T_FIXED,
    VISIBILITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Direct imports — bypass __init__.py chains that pull in optional deps
# (e.g. seaborn via src.utils.visualization).
# ---------------------------------------------------------------------------
import importlib.util as _ilu


def _import_module_from_file(name: str, path: str):
    """Import a single .py file without triggering its package __init__."""
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

normalize_landmarks_sequence = _qa.normalize_landmarks_sequence
compute_assessment_features = _qa.compute_assessment_features
_resample_sequence = _preprocess._resample_sequence

logger = logging.getLogger(__name__)


def _validate_pose_sequence(pose_sequence: list) -> np.ndarray:
    """Validate and convert the raw pose sequence to a numpy array.

    Args:
        pose_sequence: 3D list from the request body (frames × 33 × 4).

    Returns:
        np.ndarray of shape (N, 33, 4).

    Raises:
        ValueError: If the shape is invalid.
    """
    arr = np.asarray(pose_sequence, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(
            f"pose_sequence must be 3-dimensional (frames × 33 × 4), "
            f"got shape {arr.shape}."
        )
    if arr.shape[1] != 33:
        raise ValueError(
            f"Expected 33 landmarks per frame, got {arr.shape[1]}."
        )
    if arr.shape[2] != 4:
        raise ValueError(
            f"Expected 4 values per landmark (x, y, z, visibility), "
            f"got {arr.shape[2]}."
        )

    return arr


def _filter_low_visibility_frames(
    landmarks_4d: np.ndarray,
    threshold: float = VISIBILITY_THRESHOLD,
) -> np.ndarray:
    """Remove frames where mean landmark visibility is below *threshold*.

    Args:
        landmarks_4d: Shape (N, 33, 4) with [x, y, z, visibility].
        threshold: Minimum mean visibility per frame.

    Returns:
        Filtered array with the same number of columns.
    """
    mean_vis = landmarks_4d[:, :, 3].mean(axis=1)  # (N,)
    mask = mean_vis >= threshold
    filtered = landmarks_4d[mask]
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        logger.info(
            "Dropped %d / %d frames with mean visibility < %.2f",
            n_dropped, landmarks_4d.shape[0], threshold,
        )
    return filtered


def preprocess_pose_sequence(
    pose_sequence: list,
    view: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full Stage-1 preprocessing: raw mobile poses → (50, 37) features.

    Args:
        pose_sequence: 3D list/array (frames × 33 × 4).
        view: ``"front"`` or ``"side"``.

    Returns:
        Tuple of:
          - features_50: np.ndarray of shape (T_FIXED, FEATURE_DIM) — resampled
            temporal features ready for recognition / assessment.
          - lm_xyz: np.ndarray of shape (N, 33, 3) — the visibility-filtered,
            **un-normalized** 3D landmarks.  Passed downstream to
            ``segment_reps_from_sequence`` (which normalizes internally).

    Raises:
        ValueError: On shape validation or insufficient frames.
    """
    if view not in ("front", "side"):
        raise ValueError(f"view must be 'front' or 'side', got '{view}'.")

    # 1. Validate
    lm_4d = _validate_pose_sequence(pose_sequence)
    logger.info("Input frames: %d", lm_4d.shape[0])

    # 2. Filter low-visibility frames
    lm_4d = _filter_low_visibility_frames(lm_4d)

    if lm_4d.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"Too few usable frames ({lm_4d.shape[0]}). "
            f"Need at least {MIN_FRAMES}. Please record for at least 10 seconds."
        )

    # 3. Drop visibility → (N, 33, 3)
    lm_xyz = lm_4d[:, :, :3].copy()

    # 4. Normalize landmarks (pelvis-centered, torso-length-scaled)
    lm_normed = normalize_landmarks_sequence(lm_xyz)

    # 5. Compute 37 features (19 base + 18 view-specialized)
    features = compute_assessment_features(
        lm_normed, view=view, feature_type="all_extended"
    )  # (N, 37)
    assert features.shape[1] == FEATURE_DIM, (
        f"Feature dim mismatch: got {features.shape[1]}, expected {FEATURE_DIM}"
    )

    # 6. Resample to T=50
    features_50 = _resample_sequence(features, T_FIXED)  # (50, 37)

    logger.info(
        "Preprocessing complete: %d frames → (%d, %d) features",
        lm_xyz.shape[0], features_50.shape[0], features_50.shape[1],
    )

    return features_50, lm_xyz
