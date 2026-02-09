"""
Configuration constants for the AI Virtual Coach FastAPI backend.

Centralizes model paths, exercise mappings, feature dimensions, and
environment variable loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# NOTE: The exercise recognition model folder has a typo ("reognition").
RECOGNITION_MODELS_DIR = PROJECT_ROOT / "src" / "models" / "exercise_reognition_models"
ASSESSMENT_MODELS_BASE_DIR = PROJECT_ROOT / "src" / "models" / "assessment_models"

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
FEATURE_DIM: int = 37          # 19 base + 18 view-specialized
T_FIXED: int = 50              # Temporal resampling target length
MIN_FRAMES: int = 30           # Minimum frames to attempt analysis
VISIBILITY_THRESHOLD: float = 0.5  # Drop frames below this mean visibility

# ---------------------------------------------------------------------------
# Exercise mappings
# ---------------------------------------------------------------------------
# 1-indexed EXERCISE_MAP (used by assessment & coaching agent)
EXERCISE_MAP: dict[int, str] = {
    1: "Dumbbell shoulder press",
    2: "Hummer curls",
    3: "Standing Dumbbell Front Raises",
    4: "Lateral Raises",
    5: "Bulgarian split squat",
    6: "EZ Bar Curls",
    7: "Inclined Dumbbell Bench Press",
    8: "Overhead Triceps Extension",
    9: "Shrugs",
    10: "Weighted Sqauts",
    11: "Seated biceps curls",
    12: "Triceps Kickbacks",
    13: "Rows",
    14: "Deadlift",
    15: "Calf raises",
}

# Reverse: name → 1-indexed ID
EXERCISE_NAME_TO_ID: dict[str, int] = {v: k for k, v in EXERCISE_MAP.items()}

# 0-indexed int_to_label derived from *sorted* unique exercise names
# (matches `to_int()` in src/data/preprocessing.py, which is used during
# recognition model training).  Hardcoded here per option (A).
INT_TO_LABEL: dict[int, str] = {
    0:  "Bulgarian split squat",
    1:  "Calf raises",
    2:  "Deadlift",
    3:  "Dumbbell shoulder press",
    4:  "EZ Bar Curls",
    5:  "Hummer curls",
    6:  "Inclined Dumbbell Bench Press",
    7:  "Lateral Raises",
    8:  "Overhead Triceps Extension",
    9:  "Rows",
    10: "Seated biceps curls",
    11: "Shrugs",
    12: "Standing Dumbbell Front Raises",
    13: "Triceps Kickbacks",
    14: "Weighted Sqauts",
}

LABEL_TO_INT: dict[str, int] = {v: k for k, v in INT_TO_LABEL.items()}

NUM_CLASSES: int = len(INT_TO_LABEL)

# ---------------------------------------------------------------------------
# Coaching / LLM
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Inference thresholds
# ---------------------------------------------------------------------------
RECOGNITION_CONFIDENCE_THRESHOLD: float = 0.3
