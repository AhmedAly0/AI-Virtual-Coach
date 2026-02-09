"""
Quality Assessment Pipeline (37-Feature) — Exercise Video → Per-Aspect Scores.

This is the **single authoritative module** for the exercise quality assessment
pipeline. It contains all utilities previously split across ``vc_core.py`` and
``video_to_assessment_cnn_all.py``:

  - Annotation loading (weighted C1/C2 blending)
  - Landmark normalization (pelvis-centered, torso-length-scaled)
  - 37-feature computation (19 base + 18 view-specialized, in degrees)
  - Rep segmentation (exercise/view-specific 1D signal detection)
  - Temporal CNN model (``CNNSubjectRegressor``)
  - Training mode (``--train``) from NPZ data
  - Inference mode from live video via Tasks API

Usage (inference):
    python quality_assessment.py --video path/to/video.mp4 \\
        --exercise "Dumbbell shoulder press" --view front \\
        --models_dir src/models/assessment_models_37feat/

Usage (training):
    python quality_assessment.py --train \\
        --npz_path datasets/Mediapipe\\ pose\\ estimates/pose_data_front_19_features.npz \\
        --view front \\
        --annotation_dir datasets/Clips/ \\
        --out_dir src/models/assessment_models_37feat/

See README.md §8.4 for full documentation.
"""

import os
import re
import sys
import json
import logging
import warnings
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

# ---------------------------------------------------------------------------
# Import 37-feature computation from the preprocessing module.
# These functions expect normalized (33, 3) landmarks (pelvis-centered,
# torso-length-scaled) and return features in DEGREES.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.preprocess_pose_RGB import (
    # Base 19 features (13 angles + 6 distances)
    compute_all_features_from_landmarks,
    compute_angles_from_landmarks,
    compute_distances_from_landmarks,
    # Front-view specialized 18 features
    compute_specialized_features,
    # Side-view specialized 18 features
    compute_side_specialized_features,
    # Feature name constants
    ALL_FEATURE_NAMES,
    ANGLE_NAMES,
    DISTANCE_NAMES,
    SPECIALIZED_FEATURE_NAMES,
    SIDE_SPECIALIZED_FEATURE_NAMES,
    ALL_EXTENDED_FEATURE_NAMES,
    SIDE_ALL_EXTENDED_FEATURE_NAMES,
    # Landmark indices (used for rep signal generation)
    LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
    LM_LEFT_ELBOW, LM_RIGHT_ELBOW,
    LM_LEFT_WRIST, LM_RIGHT_WRIST,
    LM_LEFT_HIP, LM_RIGHT_HIP,
    LM_LEFT_KNEE, LM_RIGHT_KNEE,
    LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
    LM_LEFT_HEEL, LM_RIGHT_HEEL,
    # Angle computation helper
    calculate_angle_from_coords,
    # Resampling
    _resample_sequence,
)
from src.data.data_loader import load_pose_enhanced_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_DIM = 37  # 19 base + 18 specialized
LEGACY_FEATURE_DIM = 9
DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / "src" / "models" / "assessment_models_37feat")

# Map exercise ID → display_name
EXERCISE_MAP = {
    1: "Dumbbell shoulder press",
    2: "Hummer curls",
    3: "Standing Dumbbell Front Raises",
    4: "Lateral Raises",
    5: "Bulgarian split squat",
    6: "EZ Bar Curls",
    7: "Inclined Dumbbell Bench Press",
    8: "Overhead Triceps Extension",
    9: "Shrugs",
    10: "Weighted Squats",
    11: "Seated biceps curls",
    12: "Triceps Kickbacks",
    13: "Rows",
    14: "Deadlift",
    15: "Calf raises",
}


# ============================================================================
# §1  Small Utilities
# ============================================================================

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_")

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_vid(v):
    """Converts 'V03', 'v3', '03', 3 -> 3"""
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    digits = re.findall(r"\d+", str(v))
    if not digits:
        return None
    return int(digits[0])

def infer_volunteer_col(df):
    for c in df.columns:
        nc = norm_col(c)
        if any(k in nc for k in ["volunteer", "subject", "participant", "person", "id"]):
            return c
    return None


# ============================================================================
# §2  Annotation Loader  (0.75 × C2 + 0.25 × C1)
# ============================================================================

def load_weighted_annotation_for_exercise(exercise, annotation_dir, min_sim=0.60):
    """Load and merge two-coach annotations for an exercise from .xlsx files.

    Looks for the best-matching annotation file, detects optional C1/C2 coach
    columns, and returns weighted labels per subject/volunteer.

    Returns:
        (labels, aspect_cols, best_file, best_score)
    """
    ex_norm = norm_col(exercise)

    best_file, best_score = None, 0.0
    for f in os.listdir(annotation_dir):
        if not f.lower().endswith(".xlsx"):
            continue
        name = re.sub(r"^\d+\)\s*", "", f)
        name_norm = norm_col(name.replace(".xlsx", ""))
        sc = similarity(ex_norm, name_norm)
        if sc > best_score:
            best_score = sc
            best_file = f

    if best_file is None or best_score < min_sim:
        raise FileNotFoundError(
            f"No annotation file matches '{exercise}' (best similarity={best_score:.2f})"
        )

    path = os.path.join(annotation_dir, best_file)
    df = pd.read_excel(path)

    vol_col = infer_volunteer_col(df)
    if vol_col is None:
        raise ValueError(f"Volunteer/Subject column not found in {best_file}")

    num_cols = []
    for c in df.columns:
        if c == vol_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            num_cols.append(c)

    if not num_cols:
        raise ValueError(f"No numeric columns found in {best_file}")

    # Detect coach tag
    def coach_tag(col):
        nc = norm_col(col)
        if "c1" in nc or "coach1" in nc or "coach_1" in nc:
            return "c1"
        if "c2" in nc or "coach2" in nc or "coach_2" in nc:
            return "c2"
        return None

    def base_aspect(col):
        nc = norm_col(col)
        nc = re.sub(r"(^|_)c1(_|$)", "_", nc)
        nc = re.sub(r"(^|_)c2(_|$)", "_", nc)
        nc = nc.replace("coach1", "").replace("coach2", "")
        nc = re.sub(r"_+", "_", nc).strip("_")
        return nc

    groups = {}
    singles = []
    for c in num_cols:
        tag = coach_tag(c)
        base = base_aspect(c)
        if tag is None:
            singles.append(c)
        else:
            groups.setdefault(base, {})[tag] = c

    # If sheet already has single aspect columns (no coach split)
    if len(groups) == 0:
        aspect_cols = singles
        labels = {}
        for _, row in df.iterrows():
            vid = normalize_vid(row[vol_col])
            if vid is None:
                continue
            labels[int(vid)] = row[aspect_cols].to_numpy(np.float32)
        return labels, aspect_cols, best_file, best_score

    # Weighted per base aspect
    base_keys = sorted(groups.keys())
    aspect_cols = base_keys

    labels = {}
    for _, row in df.iterrows():
        vid = normalize_vid(row[vol_col])
        if vid is None:
            continue

        vals = []
        for base in base_keys:
            cols = groups[base]
            c1 = pd.to_numeric(row.get(cols.get("c1", None), np.nan), errors="coerce")
            c2 = pd.to_numeric(row.get(cols.get("c2", None), np.nan), errors="coerce")

            if pd.isna(c1) and pd.isna(c2):
                v = np.nan
            elif pd.isna(c1):
                v = float(c2)
            elif pd.isna(c2):
                v = float(c1)
            else:
                v = 0.25 * float(c1) + 0.75 * float(c2)

            vals.append(v)

        arr = np.array(vals, dtype=np.float32)
        if np.isnan(arr).any():
            continue
        labels[int(vid)] = arr

    return labels, aspect_cols, best_file, best_score


def load_annotations_from_workbook(exercise, workbook_path, min_sim=0.60):
    """Load annotations from a multi-sheet Excel workbook.

    Each sheet represents one exercise (e.g. ``1) Dumbbell shoulder press``).
    Rows alternate between Coach C1 and C2 for each volunteer.
    Weighting: 0.25 × C1 + 0.75 × C2.

    Returns:
        (labels_dict, aspect_cols, sheet_name, similarity_score)
    """
    xl = pd.ExcelFile(workbook_path)
    ex_norm = norm_col(exercise)

    best_sheet, best_score = None, 0.0
    for sheet in xl.sheet_names:
        name = re.sub(r"^\d+\)\s*", "", sheet)
        name_norm = norm_col(name)
        sc = similarity(ex_norm, name_norm)
        if sc > best_score:
            best_score = sc
            best_sheet = sheet

    if best_sheet is None or best_score < min_sim:
        raise FileNotFoundError(
            f"No sheet matches '{exercise}' in {workbook_path} "
            f"(best similarity={best_score:.2f})"
        )

    df = pd.read_excel(workbook_path, sheet_name=best_sheet)

    vol_col = infer_volunteer_col(df)
    if vol_col is None:
        raise ValueError(f"Volunteer column not found in sheet '{best_sheet}'")

    # Detect coach column
    coach_col = None
    for c in df.columns:
        nc = norm_col(c)
        if nc in ("coach", "coach_id", "rater", "evaluator"):
            coach_col = c
            break

    # Aspect columns = numeric columns except volunteer and coach
    aspect_cols = []
    for c in df.columns:
        if c == vol_col or c == coach_col:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            aspect_cols.append(c)

    if not aspect_cols:
        raise ValueError(f"No numeric aspect columns in sheet '{best_sheet}'")

    labels = {}

    if coach_col is not None:
        # Row-based C1/C2 format: group by volunteer
        for vol_val, group in df.groupby(vol_col):
            vid = normalize_vid(vol_val)
            if vid is None:
                continue

            coach_vals = group[coach_col].astype(str).str.strip().str.upper()
            c1_rows = group[coach_vals == "C1"]
            c2_rows = group[coach_vals == "C2"]

            c1_scores = (
                c1_rows[aspect_cols].apply(pd.to_numeric, errors="coerce")
                .mean().values
                if len(c1_rows) > 0 else None
            )
            c2_scores = (
                c2_rows[aspect_cols].apply(pd.to_numeric, errors="coerce")
                .mean().values
                if len(c2_rows) > 0 else None
            )

            if c1_scores is not None and c2_scores is not None:
                weighted = 0.25 * c1_scores + 0.75 * c2_scores
            elif c2_scores is not None:
                weighted = c2_scores
            elif c1_scores is not None:
                weighted = c1_scores
            else:
                continue

            arr = weighted.astype(np.float32)
            if np.isnan(arr).any():
                continue
            labels[vid] = arr
    else:
        # No coach column — direct scores
        for _, row in df.iterrows():
            vid = normalize_vid(row[vol_col])
            if vid is None:
                continue
            vals = pd.to_numeric(
                row[aspect_cols], errors="coerce"
            ).values.astype(np.float32)
            if np.isnan(vals).any():
                continue
            labels[vid] = vals

    aspect_names = [c.strip() for c in aspect_cols]
    return labels, aspect_names, best_sheet, best_score


# ============================================================================
# §3  Landmark Normalization  (pelvis-centered, torso-length-scaled)
# ============================================================================

# Short aliases for landmark indices
LS, RS = LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER
LE, RE = LM_LEFT_ELBOW, LM_RIGHT_ELBOW
LW, RW = LM_LEFT_WRIST, LM_RIGHT_WRIST
LH, RH = LM_LEFT_HIP, LM_RIGHT_HIP
LK, RK = LM_LEFT_KNEE, LM_RIGHT_KNEE
LA, RA = LM_LEFT_ANKLE, LM_RIGHT_ANKLE
LHEEL, RHEEL = LM_LEFT_HEEL, LM_RIGHT_HEEL


def normalize_landmarks_array(lm_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Normalize raw (33, 3) landmarks: pelvis-centered, torso-length-scaled.

    Returns:
        np.ndarray shape (33, 3) normalized, or None if torso length ≈ 0.
    """
    try:
        pelvis = (lm_xyz[LH] + lm_xyz[RH]) / 2.0
        mid_shoulder = (lm_xyz[LS] + lm_xyz[RS]) / 2.0
        torso_length = np.linalg.norm(mid_shoulder - pelvis)
        if torso_length < 1e-6:
            return None
        return (lm_xyz - pelvis) / torso_length
    except Exception:
        return None


def normalize_landmarks_sequence(lm_seq: np.ndarray) -> np.ndarray:
    """Normalize a full sequence (N, 33, 3) of raw landmarks.

    Frames that fail normalization are filled from the last good frame.
    """
    N = lm_seq.shape[0]
    out = np.zeros_like(lm_seq)
    last_good = None
    for i in range(N):
        normed = normalize_landmarks_array(lm_seq[i])
        if normed is not None:
            out[i] = normed
            last_good = normed
        elif last_good is not None:
            out[i] = last_good
    return out


# ============================================================================
# §4  Feature Computation  (37-feature unified entry point)
# ============================================================================

def compute_assessment_features(
    lm_seq_xyz: np.ndarray,
    view: str,
    feature_type: str = "all_extended",
) -> np.ndarray:
    """Compute assessment features from a normalized landmark sequence.

    Args:
        lm_seq_xyz: Shape (N_frames, 33, 3) **normalized** landmarks.
        view: 'front' or 'side'.
        feature_type: 'all_extended' (37) or 'base' (19).

    Returns:
        np.ndarray: Shape (N_frames, D) where D=37 for 'all_extended', 19 for 'base'.
    """
    if view not in ("front", "side"):
        raise ValueError(f"Invalid view '{view}'. Must be 'front' or 'side'.")

    N = lm_seq_xyz.shape[0]

    if feature_type == "base":
        feats = np.zeros((N, 19), dtype=np.float32)
        for i in range(N):
            feats[i] = compute_all_features_from_landmarks(lm_seq_xyz[i])
        return feats

    if feature_type in ("all_extended", "base_specialized"):
        feats = np.zeros((N, 37), dtype=np.float32)
        for i in range(N):
            base = compute_all_features_from_landmarks(lm_seq_xyz[i])
            if view == "front":
                spec = compute_specialized_features(lm_seq_xyz[i])
            else:
                spec = compute_side_specialized_features(lm_seq_xyz[i])
            feats[i] = np.concatenate([base, spec])
        return feats

    raise ValueError(
        f"Invalid feature_type '{feature_type}'. "
        "Use 'all_extended' (37) or 'base' (19)."
    )


def get_assessment_feature_names(view: str, feature_type: str = "all_extended"):
    """Return ordered feature name list matching ``compute_assessment_features`` output."""
    if feature_type == "base":
        return list(ALL_FEATURE_NAMES)
    if view == "front":
        return list(ALL_EXTENDED_FEATURE_NAMES)
    return list(SIDE_ALL_EXTENDED_FEATURE_NAMES)


def validate_feature_dimensions(features: np.ndarray, expected_dim: int = 37):
    """Raise ``ValueError`` if feature last dimension does not match expected."""
    actual_dim = features.shape[-1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}. "
            f"Ensure you are using the correct model checkpoint (9-feat vs 37-feat)."
        )


# ============================================================================
# §5  Legacy 9-Feature Extraction  (DEPRECATED — radians on raw landmarks)
# ============================================================================

def extract_9_features(lm_xyz):
    """Compute 9 legacy features from a single raw (33, 3) landmark frame.

    .. deprecated:: 2.0
        Use :func:`compute_assessment_features` with ``feature_type='all_extended'``
        for the 37-feature representation on normalized landmarks.
    """
    warnings.warn(
        "extract_9_features() is deprecated. Use compute_assessment_features() "
        "with feature_type='all_extended' for the 37-feature representation.",
        DeprecationWarning,
        stacklevel=2,
    )

    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))  # radians

    LS_p, RS_p = lm_xyz[LS], lm_xyz[RS]
    LE_p, RE_p = lm_xyz[LE], lm_xyz[RE]
    LW_p, RW_p = lm_xyz[LW], lm_xyz[RW]
    LH_p, RH_p = lm_xyz[LH], lm_xyz[RH]

    feats = [
        angle(LE_p, LS_p, LH_p),
        angle(RE_p, RS_p, RH_p),
        angle(LW_p, LE_p, LS_p),
        angle(RW_p, RE_p, RS_p),
        LW_p[1] - LS_p[1],
        RW_p[1] - RS_p[1],
        abs(LS_p[1] - RS_p[1]),
        abs(LW_p[1] - RW_p[1]),
        abs(LE_p[1] - RE_p[1]),
    ]
    return np.array(feats, dtype=np.float32)


# ============================================================================
# §6  YAML-Based Exercise Parameter Configuration
# ============================================================================

_CONFIG_DIR = _PROJECT_ROOT / "config"
_REP_SEG_CONFIG_PATH = _CONFIG_DIR / "rep_segmentation.yaml"

def _load_rep_segmentation_config(config_path: Optional[Path] = None) -> dict:
    """Load rep segmentation parameters from YAML config."""
    path = config_path or _REP_SEG_CONFIG_PATH
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

_REP_SEG_CONFIG: Optional[dict] = None

def _get_rep_seg_config() -> dict:
    """Lazy-load and cache the YAML config."""
    global _REP_SEG_CONFIG
    if _REP_SEG_CONFIG is None:
        _REP_SEG_CONFIG = _load_rep_segmentation_config()
    return _REP_SEG_CONFIG


# ============================================================================
# §7  Rep Signal & Segmentation Primitives
# ============================================================================

def smooth(x, k=7):
    k = int(max(1, k))
    if k == 1:
        return x
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")


def detect_peaks(signal, mode="max", thresh_pct=75, min_dist=20):
    s = smooth(np.asarray(signal, dtype=np.float32), k=7)
    if len(s) < (2 * min_dist + 5):
        return []

    if mode == "max":
        thr = np.percentile(s, thresh_pct)
        cond = lambda i: (s[i] > thr and s[i] > s[i-1] and s[i] > s[i+1])
    else:
        thr = np.percentile(s, 100 - thresh_pct)
        cond = lambda i: (s[i] < thr and s[i] < s[i-1] and s[i] < s[i+1])

    peaks = []
    for i in range(min_dist, len(s) - min_dist):
        if cond(i):
            if not peaks or (i - peaks[-1] > min_dist):
                peaks.append(i)
    return peaks


def _mean2(a, b):
    return 0.5 * (a + b)


def detect_reps_threshold_updownup(
    signal,
    up_is_high: bool = True,
    p_low: float = 20.0,
    p_high: float = 80.0,
    thr: float = None,
    smooth_k: int = 9,
    debounce: int = 3,
    extreme_mode: str = "min",
):
    """Threshold UP→DOWN→UP rep detection with debouncing.

    Returns:
        peaks: list[int] rep centers (frame indices) at the extreme inside each cycle
        debug: dict
    """
    s = smooth(np.asarray(signal, dtype=np.float32), k=int(max(1, smooth_k))).astype(np.float32)
    n = len(s)
    if n < 10:
        return [], {"method": "threshold", "reason": "too_short", "n": int(n)}

    lo = float(np.percentile(s, p_low))
    hi = float(np.percentile(s, p_high))
    if thr is None:
        thr = 0.5 * (lo + hi)
    thr = float(thr)

    if up_is_high:
        is_up = s >= thr
    else:
        is_up = s <= thr

    # Debounce: compress frames into stable states
    state = np.full(n, -1, dtype=np.int8)
    cur = 1 if is_up[0] else 0
    state[0] = cur
    for i in range(1, n):
        new = 1 if is_up[i] else 0
        if new != cur:
            ok = True
            for k in range(1, debounce):
                if i + k >= n:
                    ok = False
                    break
                if (1 if is_up[i + k] else 0) != new:
                    ok = False
                    break
            if ok:
                cur = new
        state[i] = cur

    changes = np.where(state[1:] != state[:-1])[0] + 1
    seg_starts = np.concatenate([[0], changes])
    seg_ends = np.concatenate([changes - 1, [n - 1]])
    seg_states = state[seg_starts]

    peaks = []
    i = 0
    while i + 2 < len(seg_states):
        if seg_states[i] == 1 and seg_states[i + 1] == 0 and seg_states[i + 2] == 1:
            a = int(seg_ends[i])
            b = int(seg_starts[i + 2])
            if b <= a:
                i += 1
                continue
            seg = s[a:b + 1]
            if extreme_mode == "max":
                pk = int(a + np.argmax(seg))
            else:
                pk = int(a + np.argmin(seg))
            peaks.append(pk)
            i += 2
        else:
            i += 1

    debug = {
        "method": "threshold",
        "thr": thr,
        "p_low": float(p_low),
        "p_high": float(p_high),
        "lo": lo,
        "hi": hi,
        "up_is_high": bool(up_is_high),
        "smooth_k": int(smooth_k),
        "debounce": int(debounce),
        "extreme_mode": str(extreme_mode),
        "n_peaks": int(len(peaks)),
        "signal_stats": {"min": float(np.min(s)), "max": float(np.max(s)), "mean": float(np.mean(s))},
    }
    return peaks, debug


def _pick_side_by_motion(sig_L, sig_R):
    """Choose the arm signal with higher std (more motion)."""
    return sig_L if np.std(sig_L) >= np.std(sig_R) else sig_R


def detect_peaks_k(signal, mode="max", thresh_pct=75, min_dist=20, smooth_k=7):
    """Peak detector with configurable smoothing."""
    s = smooth(np.asarray(signal, dtype=np.float32), k=int(smooth_k))
    if len(s) < (2 * int(min_dist) + 5):
        return []
    if mode == "max":
        thr = np.percentile(s, float(thresh_pct))
        cond = lambda i: (s[i] > thr and s[i] > s[i-1] and s[i] > s[i+1])
    else:
        thr = np.percentile(s, 100.0 - float(thresh_pct))
        cond = lambda i: (s[i] < thr and s[i] < s[i-1] and s[i] < s[i+1])

    peaks = []
    md = int(min_dist)
    for i in range(md, len(s) - md):
        if cond(i):
            if peaks and (i - peaks[-1] < md):
                prev = peaks[-1]
                if (mode == "max" and s[i] > s[prev]) or (mode != "max" and s[i] < s[prev]):
                    peaks[-1] = i
            else:
                peaks.append(i)
    return peaks


def _arm_raise_rel_signals(lm_arr):
    """Returns (L, R) arm-raise signals relative to shoulders."""
    yLS = lm_arr[:, LS, 1]; yRS = lm_arr[:, RS, 1]
    yLW = lm_arr[:, LW, 1]; yRW = lm_arr[:, RW, 1]
    shoulder_y = _mean2(yLS, yRS)
    relL = shoulder_y - yLW
    relR = shoulder_y - yRW
    return relL.astype(np.float32), relR.astype(np.float32)


def detect_reps_alt_arms(relL, relR, thresh_pct=70, min_dist=18, smooth_k=7, enforce_alt=True):
    """Detect reps for alternating-arm movements by merging per-arm peaks."""
    relL = np.asarray(relL, dtype=np.float32)
    relR = np.asarray(relR, dtype=np.float32)

    pkL = detect_peaks_k(relL, mode="max", thresh_pct=thresh_pct, min_dist=min_dist, smooth_k=smooth_k)
    pkR = detect_peaks_k(relR, mode="max", thresh_pct=thresh_pct, min_dist=min_dist, smooth_k=smooth_k)

    events = []
    for p in pkL:
        events.append((int(p), "L", float(relL[p])))
    for p in pkR:
        events.append((int(p), "R", float(relR[p])))
    events.sort(key=lambda x: x[0])

    # Merge close events
    merged = []
    md = int(min_dist)
    for e in events:
        if not merged:
            merged.append(e)
            continue
        if e[0] - merged[-1][0] < md:
            if e[2] > merged[-1][2]:
                merged[-1] = e
        else:
            merged.append(e)

    # Softly enforce alternation
    if enforce_alt and merged:
        alt = [merged[0]]
        for e in merged[1:]:
            if e[1] == alt[-1][1]:
                if e[2] > alt[-1][2]:
                    alt[-1] = e
            else:
                alt.append(e)
        merged = alt

    peaks = [e[0] for e in merged]
    comb = np.maximum(smooth(relL, k=int(smooth_k)), smooth(relR, k=int(smooth_k))).astype(np.float32)

    dbg = {"n_L": int(len(pkL)), "n_R": int(len(pkR)), "n_merged": int(len(peaks)), "enforce_alt": bool(enforce_alt)}
    return peaks, comb, dbg


# ============================================================================
# §8  Exercise-Specific Rep Signal Builder
# ============================================================================

def build_rep_signal(lm_arr, exercise, view):
    """Build a 1D rep-detection signal from normalized landmarks.

    Uses degree-based elbow angles and normalized y-coordinates.
    """
    ex = norm_col(exercise)

    yLS = lm_arr[:, LS, 1]; yRS = lm_arr[:, RS, 1]
    yLW = lm_arr[:, LW, 1]; yRW = lm_arr[:, RW, 1]
    yLH = lm_arr[:, LH, 1]; yRH = lm_arr[:, RH, 1]
    yLA = lm_arr[:, LA, 1]; yRA = lm_arr[:, RA, 1]
    yLHEEL = lm_arr[:, LHEEL, 1]; yRHEEL = lm_arr[:, RHEEL, 1]

    shoulder_y = _mean2(yLS, yRS)
    hip_y = _mean2(yLH, yRH)

    def elbow_angle_series_deg(side="R"):
        if side == "R":
            a = lm_arr[:, RW, :]; b = lm_arr[:, RE, :]; c = lm_arr[:, RS, :]
        else:
            a = lm_arr[:, LW, :]; b = lm_arr[:, LE, :]; c = lm_arr[:, LS, :]
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    angL = elbow_angle_series_deg("L")
    angR = elbow_angle_series_deg("R")

    # --- ARM RAISES / PRESS ---
    if ex in ["lateral_raises", "standing_dumbbell_front_raises", "dumbbell_shoulder_press"]:
        relL = shoulder_y - yLW
        relR = shoulder_y - yRW
        if view == "front":
            return _mean2(relL, relR)
        return _pick_side_by_motion(relL, relR)

    # --- CURLS / EXTENSIONS / BENCH / ROWS ---
    if ex in ["hummer_curls", "ez_bar_curls", "seated_biceps_curls",
              "overhead_triceps_extension", "triceps_kickbacks",
              "inclined_dumbbell_bench_press", "rows"]:
        sigL = -angL
        sigR = -angR
        if view == "front":
            return _mean2(sigL, sigR)
        return _pick_side_by_motion(sigL, sigR)

    # --- SHRUGS ---
    if ex == "shrugs":
        return -shoulder_y

    # --- SQUATS / SPLIT SQUAT / DEADLIFT ---
    if ex in ["weighted_squats", "weighted_sqauts", "bulgarian_split_squat", "deadlift"]:
        return hip_y

    # --- CALF RAISES ---
    if ex == "calf_raises":
        heel = _mean2(yLHEEL, yRHEEL)
        ankle = _mean2(yLA, yRA)
        return -(0.7 * heel + 0.3 * ankle)

    # Fallback
    return hip_y


# ============================================================================
# §9  Exercise Parameters (hardcoded fallback, YAML preferred)
# ============================================================================

EXERCISE_PARAMS = {
    "lateral_raises": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=75, mode="max"),
    },
    "standing_dumbbell_front_raises": {
        "front": dict(method="alt_arms", win=25, min_dist=18, thresh_pct=70, mode="max", smooth_k=7, enforce_alt=True),
        "side":  dict(method="alt_arms", win=25, min_dist=18, thresh_pct=70, mode="max", smooth_k=7, enforce_alt=False),
    },
    "dumbbell_shoulder_press": {
        "front": dict(method="peaks", win=25, min_dist=25, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=25, thresh_pct=75, mode="max"),
    },
    "hummer_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "ez_bar_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "seated_biceps_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "overhead_triceps_extension": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "triceps_kickbacks": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "inclined_dumbbell_bench_press": {
        "front": dict(method="peaks", win=30, min_dist=25, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=30, min_dist=25, thresh_pct=70, mode="max"),
    },
    "rows": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "shrugs": {
        "front": dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
    },
    "weighted_squats": {
        "front": dict(method="threshold", win=38, smooth_k=11, debounce=3,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
        "side":  dict(method="threshold", win=38, smooth_k=11, debounce=4,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
    },
    "weighted_sqauts": {
        "front": dict(method="threshold", win=38, smooth_k=11, debounce=3,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
        "side":  dict(method="threshold", win=38, smooth_k=11, debounce=4,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
    },
    "bulgarian_split_squat": {
        "front": dict(method="peaks", win=30, min_dist=28, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=30, min_dist=28, thresh_pct=75, mode="max"),
    },
    "deadlift": {
        "front": dict(method="peaks", win=35, min_dist=35, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=35, min_dist=35, thresh_pct=75, mode="max"),
    },
    "calf_raises": {
        "front": dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
    },
}


def _params_for(exercise, view, win_override=None, min_dist_override=None):
    """Resolve segmentation params: YAML config → hardcoded → defaults."""
    ex = norm_col(exercise)
    v = "front" if view == "front" else "side"

    cfg = _get_rep_seg_config()
    yaml_exercises = cfg.get("exercises", {})
    yaml_defaults = cfg.get("defaults", {})
    base = None
    if ex in yaml_exercises and v in yaml_exercises[ex]:
        base = dict(yaml_exercises[ex][v])
    elif ex in EXERCISE_PARAMS and v in EXERCISE_PARAMS.get(ex, {}):
        base = dict(EXERCISE_PARAMS[ex][v])
    else:
        base = dict(yaml_defaults) if yaml_defaults else dict(win=25, min_dist=20, thresh_pct=75, mode="max")

    p = dict(base)
    if win_override is not None:
        p["win"] = int(win_override)
    if min_dist_override is not None:
        p["min_dist"] = int(min_dist_override)
    return p


# ============================================================================
# §10  Main Segmentation Entry Point
# ============================================================================

def _extract_window(feats: np.ndarray, center: int, win: int, pad_edges: bool):
    """Extract a (2*win, F) feature window around center, optionally padding edges."""
    N = feats.shape[0]
    a = int(center) - int(win)
    b = int(center) + int(win)
    if not pad_edges:
        if a < 0 or b > N:
            return None
        return feats[a:b]

    out = np.empty((2 * win, feats.shape[1]), dtype=np.float32)
    for i in range(2 * win):
        t = a + i
        if t < 0:
            out[i] = feats[0]
        elif t >= N:
            out[i] = feats[-1]
        else:
            out[i] = feats[t]
    return out


def _quality_score(signal: np.ndarray, center: int, win: int):
    """Local amplitude inside the window around 'center'."""
    a = max(0, int(center) - int(win))
    b = min(len(signal), int(center) + int(win))
    seg = signal[a:b]
    if len(seg) < 3:
        return 0.0
    return float(np.max(seg) - np.min(seg))


def _filter_peaks(peaks, signal, win, max_reps=40, q_drop_pct=0):
    """Filter peaks by amplitude quality and cap max reps."""
    peaks = list(map(int, peaks))
    if len(peaks) == 0:
        return [], {"n_raw": 0}

    qs = np.array([_quality_score(signal, p, win) for p in peaks], dtype=np.float32)
    order = np.argsort(-qs)

    keep = order
    if q_drop_pct and len(peaks) >= 5:
        k = int(np.ceil(len(peaks) * (1.0 - q_drop_pct / 100.0)))
        k = max(1, k)
        keep = keep[:k]

    if max_reps and len(keep) > max_reps:
        keep = keep[:max_reps]

    kept = sorted([peaks[i] for i in keep])
    dbg = {
        "n_raw": int(len(peaks)),
        "n_kept": int(len(kept)),
        "q_min": float(np.min(qs)) if len(qs) else None,
        "q_max": float(np.max(qs)) if len(qs) else None,
        "q_mean": float(np.mean(qs)) if len(qs) else None,
        "q_drop_pct": int(q_drop_pct),
        "max_reps": int(max_reps) if max_reps else None,
    }
    return kept, dbg


def segment_reps_from_sequence(
    exercise,
    view,
    lm_seq_xyz,
    win=25,
    min_dist=20,
    thresh_pct=None,
    feature_mode=None,
    **kwargs,
):
    """Segment an exercise video into reps and extract feature windows.

    Args:
        exercise: Exercise name (e.g. 'Dumbbell shoulder press').
        view: 'front' or 'side'.
        lm_seq_xyz: (N, 33, 3) landmark sequence (raw or normalized).
        win: Half-window size override (frames).
        min_dist: Minimum peak separation override (frames).
        thresh_pct: Peak threshold percentile override.
        feature_mode: '37' (default), '9' (legacy), or read from YAML.

    Returns:
        reps: (R, 2*win, D) feature windows where D=37 or 9.
        peaks: List[int] of peak frame indices.
        signal: 1D numpy array of the rep-detection signal.
        used: Dict of actual parameters and debug info.
    """
    lm_arr = np.asarray(lm_seq_xyz, dtype=np.float32)
    if lm_arr.ndim != 3 or lm_arr.shape[1] < 33 or lm_arr.shape[2] < 3:
        raise ValueError(
            "lm_seq_xyz must be (N, 33, 3) — N frames of 33 landmarks of (x,y,z)."
        )

    # Resolve feature mode
    if feature_mode is None:
        cfg = _get_rep_seg_config()
        feature_mode = cfg.get("feature_mode", os.environ.get("ASSESSMENT_FEATURE_MODE", "37"))
    feature_mode = str(feature_mode)

    if feature_mode == "9":
        warnings.warn(
            "Using legacy 9-feature mode. Switch to feature_mode='37' for best performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            feats = np.array([extract_9_features(lm_arr[i]) for i in range(lm_arr.shape[0])], dtype=np.float32)
        lm_for_signal = lm_arr
    else:
        lm_normed = normalize_landmarks_sequence(lm_arr)
        feats = compute_assessment_features(lm_normed, view=view, feature_type="all_extended")
        lm_for_signal = lm_normed

    params = _params_for(exercise, view, win_override=win, min_dist_override=min_dist)
    if thresh_pct is not None:
        params["thresh_pct"] = int(thresh_pct)

    signal = build_rep_signal(lm_for_signal, exercise, view).astype(np.float32)

    used = {"exercise": str(exercise), "view": str(view), "feature_mode": feature_mode}
    method = params.get("method", "peaks")
    used["method"] = method

    if method == "threshold":
        p_low = float(params.get("p_low", 20))
        p_high = float(params.get("p_high", 80))
        smooth_k = int(params.get("smooth_k", 9))
        debounce = int(params.get("debounce", 3))
        up_is_high = bool(params.get("up_is_high", True))
        extreme_mode = str(params.get("extreme_mode", "min"))
        pad_edges = bool(params.get("pad_edges", True))
        max_reps = int(params.get("max_reps", 40))
        q_drop_pct = int(params.get("q_drop_pct", 0))

        relax = params.get("relax", [(p_low, p_high), (max(5, p_low-5), min(95, p_high+5))])

        debug_passes = []
        peaks = []
        chosen = None
        for pl, ph in relax:
            pk, dbg = detect_reps_threshold_updownup(
                signal,
                up_is_high=up_is_high,
                p_low=float(pl),
                p_high=float(ph),
                smooth_k=smooth_k,
                debounce=debounce,
                extreme_mode=extreme_mode,
            )
            debug_passes.append({"p_low": float(pl), "p_high": float(ph), "n_peaks": int(len(pk)), "dbg": dbg})
            if len(pk) >= 1:
                peaks = pk
                chosen = (pl, ph)
                break
        used["threshold_debug_passes"] = debug_passes
        if chosen is not None:
            used["p_low_used"] = float(chosen[0])
            used["p_high_used"] = float(chosen[1])

        w = int(params.get("win", win))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges
        used["win"] = int(w)

    elif method == "alt_arms":
        w = int(params.get("win", win))
        tp = int(params.get("thresh_pct", 70))
        md = int(params.get("min_dist", min_dist))
        smooth_k = int(params.get("smooth_k", 7))
        enforce_alt = bool(params.get("enforce_alt", True))

        relL, relR = _arm_raise_rel_signals(lm_for_signal)
        peaks, sig_alt, dbg = detect_reps_alt_arms(
            relL, relR, thresh_pct=tp, min_dist=md, smooth_k=smooth_k, enforce_alt=enforce_alt
        )
        signal = sig_alt.astype(np.float32)

        pad_edges = bool(params.get("pad_edges", False))
        max_reps = int(params.get("max_reps", 60))
        q_drop_pct = int(params.get("q_drop_pct", 0))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)

        used.update({
            "win": int(w), "mode": "max", "thresh_pct": int(tp), "min_dist": int(md),
            "smooth_k": int(smooth_k), "enforce_alt": bool(enforce_alt),
        })
        used["alt_arms_debug"] = dbg
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges

    else:
        # Classic peak method
        w = int(params.get("win", win))
        mode = params.get("mode", "max")
        tp = int(params.get("thresh_pct", 75))
        md = int(params.get("min_dist", min_dist))
        peaks = detect_peaks(signal, mode=mode, thresh_pct=tp, min_dist=md)

        pad_edges = bool(params.get("pad_edges", False))
        max_reps = int(params.get("max_reps", 40))
        q_drop_pct = int(params.get("q_drop_pct", 0))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)

        used.update({"win": int(w), "mode": str(mode), "thresh_pct": int(tp), "min_dist": int(md)})
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges

    # Build rep windows
    w = int(used["win"])
    reps = []
    kept_peaks = []
    for p in peaks:
        window = _extract_window(feats, p, w, pad_edges=bool(used.get("pad_edges", False)))
        if window is not None and window.shape[0] == 2 * w:
            reps.append(window)
            kept_peaks.append(int(p))

    reps = np.asarray(reps, dtype=np.float32) if len(reps) else np.zeros((0, 2*w, feats.shape[1]), dtype=np.float32)

    used["peaks"] = kept_peaks
    used["n_reps"] = int(reps.shape[0])
    used["signal_stats"] = {"min": float(np.min(signal)), "max": float(np.max(signal)), "mean": float(np.mean(signal))}
    return reps, kept_peaks, signal, used


# ============================================================================
# §11  Model Load Utilities
# ============================================================================

def _load_model_checkpoint(model_dir: str, exercise: str) -> Tuple[tf.keras.Model, dict]:
    """Load a saved Keras model and its metadata JSON.

    Looks for ``<safe_name>_best.keras`` (or ``.h5``) and
    ``<safe_name>_meta.json`` in *model_dir*.
    """
    base = safe_name(exercise)
    meta_path = os.path.join(model_dir, f"{base}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Prefer .keras, fall back to .h5
    keras_path = os.path.join(model_dir, f"{base}_best.keras")
    h5_path = os.path.join(model_dir, f"{base}_best.h5")
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path, compile=False)
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path, compile=False)
    else:
        raise FileNotFoundError(
            f"Model weights not found at {keras_path} or {h5_path}"
        )
    return model, meta


# ============================================================================
# §12  Temporal CNN Model  (TensorFlow / Keras)
# ============================================================================

def build_cnn_subject_regressor(
    in_feats: int,
    n_aspects: int,
    T_fixed: int = 50,
    name: str = "cnn_subject_regressor",
) -> tf.keras.Model:
    """Build a temporal-CNN Keras model for exercise quality assessment.

    Architecture:
        Input  (B, T, F)      — B=batch, T=timesteps, F=features
        Conv1D (64, k=5)      + ReLU
        Conv1D (128, k=5)     + ReLU
        GlobalAveragePooling1D → (B, 128)
        Dense  (128→A)        + Sigmoid → A aspect scores in [0, 1]

    Note:
        When the model is used during inference with variable-length reps,
        each rep is passed individually and attention pooling across reps is
        handled outside the model (see ``process_exercise_video``).
    """
    inp = tf.keras.Input(shape=(T_fixed, in_feats), name="features")
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(n_aspects, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name=name)


def _attention_pool_reps(rep_embeddings: np.ndarray) -> np.ndarray:
    """Soft-attention pooling across R rep embeddings → single subject vector.

    Args:
        rep_embeddings: (R, A) predicted scores per rep.

    Returns:
        (A,) weighted-average scores.
    """
    # Simple amplitude-based weighting (reps with higher overall score get
    # slightly more weight, dampened by softmax temperature).
    if rep_embeddings.shape[0] == 1:
        return rep_embeddings[0]
    energy = rep_embeddings.mean(axis=-1)  # (R,)
    weights = np.exp(energy) / np.sum(np.exp(energy))  # softmax
    return (rep_embeddings * weights[:, None]).sum(axis=0)


# ============================================================================
# §13  Training Mode
# ============================================================================

def _load_training_data_from_npz(
    npz_path: str,
    view: str,
    annotation_dir: str,
    exercises: Optional[List[str]] = None,
    T_fixed: int = 50,
) -> Dict[str, Dict]:
    """Load pre-computed features from NPZ and pair with annotations.

    The NPZ file contains pre-computed feature arrays:
      - ``X_all_features``      (N, T, 19) — base features (13 angles + 6 distances)
      - ``X_specialized``       (N, T, 18) — front-view specialized (front NPZ)
      - ``X_side_specialized``  (N, T, 18) — side-view specialized  (side NPZ)

    These are concatenated to form 37-feature vectors.  Falls back to computing
    features from ``X_landmarks`` if pre-computed arrays are absent.

    Args:
        annotation_dir: Path to a single annotation ``.xlsx`` workbook **or**
            a directory containing per-exercise ``.xlsx`` files.

    Returns:
        Dict mapping exercise_name → {X, Y, aspect_cols, subject_ids}
    """
    data = np.load(npz_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # Assemble 37 features from pre-computed arrays (or fallback)
    # ------------------------------------------------------------------
    if "X_all_features" in data:
        X_base = data["X_all_features"]  # (N, T, 19)
        if view == "side" and "X_side_specialized" in data:
            X_spec = data["X_side_specialized"]  # (N, T, 18)
        elif "X_specialized" in data:
            X_spec = data["X_specialized"]  # (N, T, 18)
        else:
            raise KeyError(
                f"NPZ missing specialized features for view='{view}'. "
                f"Available keys: {list(data.keys())}"
            )
        X_features = np.concatenate([X_base, X_spec], axis=-1)  # (N, T, 37)
        logger.info(
            "Pre-computed features: base %s + specialized %s → %s",
            X_base.shape, X_spec.shape, X_features.shape,
        )
    elif "X_landmarks" in data:
        logger.info("No pre-computed features — computing from X_landmarks …")
        X_landmarks = data["X_landmarks"]
        X_features_list = []
        for i in range(X_landmarks.shape[0]):
            feats = compute_assessment_features(
                X_landmarks[i], view=view, feature_type="all_extended"
            )
            if feats.shape[0] != T_fixed:
                feats = _resample_sequence(feats, T_fixed)
            X_features_list.append(feats)
        X_features = np.array(X_features_list, dtype=np.float32)
    else:
        raise KeyError(
            f"NPZ at {npz_path} has neither 'X_all_features' nor "
            f"'X_landmarks'. Keys: {list(data.keys())}"
        )

    exercise_names = data["exercise_names"]
    subject_ids = data["subject_ids"]

    # Ensure proper types after NPZ deserialization
    if exercise_names.dtype.kind in ("U", "S", "O"):
        exercise_names = np.array([str(e) for e in exercise_names])
    if subject_ids.dtype.kind in ("U", "S", "O"):
        subject_ids = np.array([int(s) for s in subject_ids])

    unique_exercises = sorted(set(exercise_names))
    if exercises:
        exercises_norm = [norm_col(x) for x in exercises]
        unique_exercises = [
            e for e in unique_exercises if norm_col(e) in exercises_norm
        ]

    # ------------------------------------------------------------------
    # Auto-detect annotation source (single workbook vs. directory)
    # ------------------------------------------------------------------
    ann_path = annotation_dir
    use_workbook = False

    if os.path.isfile(ann_path) and ann_path.lower().endswith(".xlsx"):
        use_workbook = True
        logger.info("Annotation source: workbook file %s", ann_path)
    elif os.path.isdir(ann_path):
        # Check if directory contains a single multi-sheet workbook
        xlsx_in_dir = [
            f for f in os.listdir(ann_path) if f.lower().endswith(".xlsx")
        ]
        if len(xlsx_in_dir) == 1:
            candidate = os.path.join(ann_path, xlsx_in_dir[0])
            try:
                xl_check = pd.ExcelFile(candidate)
                if len(xl_check.sheet_names) > 1:
                    ann_path = candidate
                    use_workbook = True
                    logger.info(
                        "Auto-detected multi-sheet workbook: %s", candidate
                    )
            except Exception:
                pass
        if not use_workbook:
            logger.info("Annotation source: per-exercise .xlsx in %s", ann_path)
    else:
        raise FileNotFoundError(f"Annotation path not found: {ann_path}")

    # ------------------------------------------------------------------
    # Pair features with annotations per exercise
    # ------------------------------------------------------------------
    result = {}
    for ex_name in unique_exercises:
        try:
            if use_workbook:
                labels, aspect_cols, ann_src, ann_score = (
                    load_annotations_from_workbook(ex_name, ann_path)
                )
            else:
                labels, aspect_cols, ann_src, ann_score = (
                    load_weighted_annotation_for_exercise(ex_name, ann_path)
                )
        except FileNotFoundError:
            logger.warning("No annotation for '%s', skipping.", ex_name)
            continue

        mask = exercise_names == ex_name
        X_ex = X_features[mask]
        sids_ex = subject_ids[mask]

        X_list, Y_list, kept_sids = [], [], []
        for i in range(X_ex.shape[0]):
            sid = int(sids_ex[i])
            if sid not in labels:
                continue

            feat_seq = X_ex[i]
            if feat_seq.shape[0] != T_fixed:
                feat_seq = _resample_sequence(feat_seq, T_fixed)

            X_list.append(feat_seq)
            Y_list.append(labels[sid])
            kept_sids.append(sid)

        if not X_list:
            logger.warning("No matched samples for '%s', skipping.", ex_name)
            continue

        result[ex_name] = {
            "X": np.array(X_list, dtype=np.float32),
            "Y": np.array(Y_list, dtype=np.float32),
            "aspect_cols": aspect_cols,
            "subject_ids": kept_sids,
        }
        logger.info(
            "  %-35s → %d samples, %d aspects (%s)",
            ex_name, len(kept_sids), len(aspect_cols), ann_src,
        )

    return result


def _subject_disjoint_split(
    subject_ids: List[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split sample indices by subject ID (no subject in multiple sets)."""
    rng = np.random.RandomState(seed)
    unique_sids = sorted(set(subject_ids))
    rng.shuffle(unique_sids)

    n = len(unique_sids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_sids = set(unique_sids[:n_test])
    val_sids = set(unique_sids[n_test : n_test + n_val])
    train_sids = set(unique_sids[n_test + n_val :])

    sid_arr = np.array(subject_ids)
    train_idx = [i for i in range(len(subject_ids)) if sid_arr[i] in train_sids]
    val_idx = [i for i in range(len(subject_ids)) if sid_arr[i] in val_sids]
    test_idx = [i for i in range(len(subject_ids)) if sid_arr[i] in test_sids]

    return train_idx, val_idx, test_idx


def train_assessment_models(
    npz_path: str,
    view: str,
    annotation_dir: str,
    out_dir: str,
    exercises: Optional[List[str]] = None,
    T_fixed: int = 50,
    max_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 8,
    patience: int = 30,
    seed: int = 42,
    device_str: str = "auto",
):
    """Train per-exercise temporal CNN assessment models with 37 features.

    Saves per exercise:
        - ``<name>_best.keras``  — Keras model weights
        - ``<name>_meta.json``   — metadata (aspect_cols, feature_dim, MAE, …)
    """
    # GPU configuration
    if device_str == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Using GPU: %s", gpus[0].name)
        else:
            logger.info("No GPU found — training on CPU.")
    elif device_str == "cpu":
        tf.config.set_visible_devices([], "GPU")

    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading training data from %s (view=%s)...", npz_path, view)
    all_data = _load_training_data_from_npz(
        npz_path, view, annotation_dir, exercises=exercises, T_fixed=T_fixed
    )

    if not all_data:
        raise RuntimeError("No training data loaded. Check NPZ and annotation paths.")

    summary = {}
    for ex_name, data in all_data.items():
        logger.info("\n" + "=" * 70)
        logger.info("Training: %s  (%d samples, %d aspects)", ex_name, data["X"].shape[0], data["Y"].shape[1])
        logger.info("=" * 70)

        X = data["X"]            # (N, T_fixed, 37)
        Y = data["Y"]            # (N, n_aspects)  in 0-10 scale
        aspect_cols = data["aspect_cols"]
        subject_ids = data["subject_ids"]
        n_aspects = Y.shape[1]

        Y_norm = Y / 10.0        # scale to [0, 1] for sigmoid output

        train_idx, val_idx, test_idx = _subject_disjoint_split(subject_ids, seed=seed)
        if not train_idx:
            logger.warning("Not enough subjects for '%s', skipping.", ex_name)
            continue

        X_train, Y_train = X[train_idx], Y_norm[train_idx]
        X_val = X[val_idx] if val_idx else X[train_idx[:1]]
        Y_val = Y_norm[val_idx] if val_idx else Y_norm[train_idx[:1]]
        X_test = X[test_idx] if test_idx else X[train_idx[:1]]
        Y_test = Y_norm[test_idx] if test_idx else Y_norm[train_idx[:1]]

        logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

        # ---- Build & compile ----
        model = build_cnn_subject_regressor(
            in_feats=FEATURE_DIM, n_aspects=n_aspects, T_fixed=T_fixed,
            name=f"assess_{safe_name(ex_name)}",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        # ---- Callbacks ----
        best_keras_path = os.path.join(out_dir, f"{safe_name(ex_name)}_best.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_keras_path,
                monitor="val_loss", save_best_only=True, verbose=0,
            ),
        ]

        # ---- Train ----
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        # ---- Evaluate on test ----
        test_pred = model.predict(X_test, verbose=0)       # (N_test, n_aspects)
        test_mae = float(np.mean(np.abs(test_pred * 10.0 - Y_test * 10.0)))
        best_val_loss = float(min(history.history["val_loss"]))

        logger.info("  ✅ %s  test_MAE=%.2f  best_val_MSE=%.4f", ex_name, test_mae, best_val_loss)

        # ---- Save metadata JSON ----
        meta = {
            "aspect_cols": list(aspect_cols) if not isinstance(aspect_cols, list) else aspect_cols,
            "feature_dim": FEATURE_DIM,
            "exercise": ex_name,
            "view": view,
            "test_mae_0_10": test_mae,
            "val_mse_0_1": best_val_loss,
            "T_fixed": T_fixed,
            "n_aspects": n_aspects,
        }
        meta_path = os.path.join(out_dir, f"{safe_name(ex_name)}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("  Saved: %s  +  %s", best_keras_path, meta_path)

        summary[ex_name] = {
            "test_mae": test_mae,
            "val_mse": best_val_loss,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "n_aspects": n_aspects,
            "aspect_cols": list(aspect_cols) if not isinstance(aspect_cols, list) else aspect_cols,
        }

    summary_path = os.path.join(out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nTraining summary saved to: %s", summary_path)

    return summary


# ============================================================================
# §14  Inference Mode — Video → Scores
# ============================================================================

def _extract_landmarks_from_video_tasks_api(
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Extract (N, 33, 3) raw landmarks from video using MediaPipe Tasks API."""
    import cv2

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "mediapipe is required. Install with: pip install mediapipe"
        )

    model_path = str(_PROJECT_ROOT / "datasets" / "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PoseLandmarker model not found at {model_path}. "
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    lm_seq = []
    last_good = None
    processed = 0
    frame_i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if stride > 1 and (frame_i % stride != 0):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            last_good = arr
            lm_seq.append(arr)
        elif last_good is not None:
            lm_seq.append(last_good.copy())

        processed += 1
        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    landmarker.close()

    if len(lm_seq) < 10:
        raise RuntimeError(
            f"Video too short or pose detection failed (only {len(lm_seq)} frames with pose)."
        )

    return np.array(lm_seq, dtype=np.float32)


def process_exercise_video(
    video_path: str,
    exercise: str,
    view: str,
    model_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    debug: bool = False,
) -> Dict:
    """Process a single exercise video through the 37-feature assessment pipeline.

    Pipeline:
        1. Extract raw landmarks via Tasks API
        2. Normalize landmarks (pelvis-centered, torso-length-scaled)
        3. Segment reps using exercise-specific signal detection
        4. Compute 37-feature windows per rep
        5. Run temporal CNN for per-aspect quality scores

    Returns:
        Dict with keys: exercise, view, n_reps, scores_0_10, etc.
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    model, meta = _load_model_checkpoint(models_dir, exercise)
    aspect_cols = meta["aspect_cols"]
    ckpt_feat_dim = int(meta.get("feature_dim", LEGACY_FEATURE_DIM))

    if ckpt_feat_dim == LEGACY_FEATURE_DIM:
        raise ValueError(
            f"Checkpoint uses legacy {LEGACY_FEATURE_DIM}-feature mode. "
            f"Please retrain with --train for 37-feature models."
        )

    if ckpt_feat_dim != FEATURE_DIM:
        raise ValueError(
            f"Feature dimension mismatch: checkpoint has {ckpt_feat_dim}, "
            f"expected {FEATURE_DIM}."
        )

    logger.info("Extracting landmarks from video...")
    lm_seq_raw = _extract_landmarks_from_video_tasks_api(
        video_path, stride=stride, max_frames=max_frames
    )

    if len(lm_seq_raw) < 50:
        raise RuntimeError(
            "Video too short or pose detection failed (too few frames with pose)."
        )

    logger.info("Segmenting reps...")
    reps, peaks, signal, used = segment_reps_from_sequence(
        exercise=exercise,
        view=view,
        lm_seq_xyz=lm_seq_raw,
        feature_mode="37",
    )

    R = int(reps.shape[0])
    if R == 0:
        raise RuntimeError(
            "No reps detected. Try adjusting parameters in config/rep_segmentation.yaml."
        )

    validate_feature_dimensions(reps, expected_dim=FEATURE_DIM)

    logger.info("Running CNN inference on %d reps...", R)
    per_rep_scores = model.predict(reps, verbose=0)  # (R, n_aspects)
    pooled = _attention_pool_reps(per_rep_scores)     # (n_aspects,)

    scores_0_10 = (pooled * 10.0).tolist()

    result = {
        "exercise": exercise,
        "view": view,
        "models_dir": models_dir,
        "feature_dim": FEATURE_DIM,
        "n_frames_used": int(len(lm_seq_raw)),
        "n_reps": R,
        "scores_0_10": {
            aspect_cols[i]: round(scores_0_10[i], 2) for i in range(len(aspect_cols))
        },
    }

    if debug:
        result["debug"] = {
            "peaks": [int(p) for p in peaks],
            "used_params": used,
            "per_rep_scores": (per_rep_scores * 10.0).tolist(),
            "signal_preview": {
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "mean": float(np.mean(signal)),
            },
        }

    return result


def process_exercise_video_37(
    video_path: str,
    view: str,
    exercise_id: int,
    model_path: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Convenience wrapper using exercise ID instead of name.

    Args:
        video_path: Path to video.
        view: 'front' or 'side'.
        exercise_id: Integer ID (1–15).
        model_path: Optional model checkpoint path or directory.
    """
    exercise = EXERCISE_MAP.get(exercise_id)
    if exercise is None:
        raise ValueError(
            f"Unknown exercise_id={exercise_id}. Valid: {list(EXERCISE_MAP.keys())}"
        )
    return process_exercise_video(
        video_path=video_path,
        exercise=exercise,
        view=view,
        model_path=model_path,
        **kwargs,
    )


# ============================================================================
# §15  CLI Entry Point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Exercise Quality Assessment (37-Feature Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on a video:
  python quality_assessment.py --video path/to/video.mp4 \\
      --exercise "Dumbbell shoulder press" --view front

  # Train models from NPZ data (single workbook):
  python quality_assessment.py --train \\
      --npz_path "datasets/Mediapipe pose estimates/pose_data_front_19_features.npz" \\
      --view front \\
      --annotation_dir "datasets/Annotation_.xlsx"

  # Train models (annotation directory):
  python quality_assessment.py --train \\
      --npz_path "datasets/Mediapipe pose estimates/pose_data_front_19_features.npz" \\
      --view front \\
      --annotation_dir "datasets/Clips/"
""",
    )

    ap.add_argument("--train", action="store_true", help="Train assessment models from NPZ data.")

    ap.add_argument("--video", help="Path to exercise video (inference mode).")
    ap.add_argument("--exercise", help="Exercise name (inference mode).")
    ap.add_argument("--view", choices=["front", "side"], help="Camera view.")
    ap.add_argument("--out", default="assessment.json", help="Output JSON path (inference mode).")
    ap.add_argument("--model", default=None, help="Direct path to .keras model (unused, kept for compat).")
    ap.add_argument("--models_dir", default=None, help="Directory of .keras models + _meta.json files.")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    ap.add_argument("--max_frames", type=int, default=None, help="Stop after N frames (debug).")
    ap.add_argument("--debug", action="store_true", help="Include debug info in output.")

    ap.add_argument("--npz_path", help="Path to NPZ file with landmarks (train mode).")
    ap.add_argument("--annotation_dir",
                    help="Path to annotation .xlsx workbook OR directory with per-exercise .xlsx files (train mode).")
    ap.add_argument("--out_dir", default=DEFAULT_MODELS_DIR, help="Output directory for trained models.")
    ap.add_argument("--max_epochs", type=int, default=200, help="Max training epochs.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    ap.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits.")

    args = ap.parse_args()

    if args.train:
        if not args.npz_path:
            ap.error("--npz_path is required in --train mode.")
        if not args.view:
            ap.error("--view is required in --train mode.")
        if not args.annotation_dir:
            ap.error("--annotation_dir is required in --train mode.")

        summary = train_assessment_models(
            npz_path=args.npz_path,
            view=args.view,
            annotation_dir=args.annotation_dir,
            out_dir=args.out_dir,
            max_epochs=args.max_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        )

        print("\n✅ Training complete")
        for ex, stats in summary.items():
            print(f"  {ex}: MAE={stats['test_mae']:.2f}, samples={stats['n_train']}+{stats['n_val']}+{stats['n_test']}")

    else:
        if not args.video:
            ap.error("--video is required in inference mode (or use --train).")
        if not args.exercise:
            ap.error("--exercise is required in inference mode.")
        if not args.view:
            ap.error("--view is required in inference mode.")

        result = process_exercise_video(
            video_path=args.video,
            exercise=args.exercise,
            view=args.view,
            model_path=args.model,
            models_dir=args.models_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            debug=args.debug,
        )

        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

        print("\n✅ Assessment complete")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
