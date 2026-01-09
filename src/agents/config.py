"""
Configuration file for AI agents.

Loads configuration from environment variables with sensible defaults.
API keys should be set in .env file (not committed to version control).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in project root
_project_root = Path(__file__).parent.parent.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path)

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY not set. Please set it in your .env file or environment. "
        "See .env.example for reference."
    )

# Model configuration
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Agent settings
DEFAULT_USE_LLM = True
MAX_FEEDBACK_WORDS = 100

# Score thresholds for feedback categorization
SCORE_THRESHOLDS = {
    "excellent": 8.5,
    "good": 7.0,
    "needs_improvement": 5.0,
    "poor": 0.0,
}
