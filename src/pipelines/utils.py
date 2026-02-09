"""
Shared utilities for the FastAPI backend pipeline.

- Fallback feedback generator (when Gemini LLM is unavailable)
- Startup model pre-loading helper
"""

import logging
from typing import List

from .config import EXERCISE_MAP, EXERCISE_NAME_TO_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback feedback (no LLM)
# ---------------------------------------------------------------------------

def generate_fallback_feedback(
    per_rep_scores: list[dict],
    aspect_means: dict[str, float],
    overall_score: float,
) -> list[str]:
    """Produce rule-based feedback strings when the Gemini LLM is unavailable.

    Args:
        per_rep_scores: List of per-rep score dicts.
        aspect_means: Mean per-aspect scores across all reps.
        overall_score: Single overall score (0-10).

    Returns:
        List of human-readable feedback strings.
    """
    tips: list[str] = []

    # Overall summary
    if overall_score >= 8.5:
        tips.append(
            f"Excellent form! Your overall score was {overall_score:.1f}/10. Keep it up!"
        )
    elif overall_score >= 7.0:
        tips.append(
            f"Good form overall ({overall_score:.1f}/10). A few areas to refine."
        )
    elif overall_score >= 5.0:
        tips.append(
            f"Your overall score was {overall_score:.1f}/10 — there's room for improvement."
        )
    else:
        tips.append(
            f"Your form needs attention (overall {overall_score:.1f}/10). "
            "Consider reviewing proper technique or lowering the weight."
        )

    # Identify weakest and strongest aspects
    if aspect_means:
        sorted_aspects = sorted(aspect_means.items(), key=lambda x: x[1])
        weakest_name, weakest_score = sorted_aspects[0]
        strongest_name, strongest_score = sorted_aspects[-1]

        tips.append(
            f"Strongest aspect: {strongest_name} ({strongest_score:.1f}/10)."
        )
        if weakest_score < 7.0:
            tips.append(
                f"Focus on improving: {weakest_name} ({weakest_score:.1f}/10)."
            )

    # Fatigue check — compare first 3 vs last 3 reps
    if len(per_rep_scores) >= 6:
        early_avg = _mean_rep_score(per_rep_scores[:3])
        late_avg = _mean_rep_score(per_rep_scores[-3:])
        if early_avg - late_avg > 0.8:
            tips.append(
                f"Fatigue detected: your form dropped from {early_avg:.1f} "
                f"(early reps) to {late_avg:.1f} (final reps). "
                "Consider reducing weight or taking longer rest."
            )

    return tips


def _mean_rep_score(reps: list[dict]) -> float:
    """Average all aspect scores across a slice of reps."""
    all_vals = []
    for rep in reps:
        all_vals.extend(rep["scores"].values())
    return sum(all_vals) / max(len(all_vals), 1)


# ---------------------------------------------------------------------------
# Startup pre-loader
# ---------------------------------------------------------------------------

def preload_models(views: list[str] | None = None) -> None:
    """Eagerly load recognition models at server startup.

    Assessment models are loaded lazily on first request per exercise
    because there are 15 × 2 = 30 of them.

    Args:
        views: List of views to preload (default: ``["front", "side"]``).
    """
    from .recognition import load_recognition_model

    views = views or ["front", "side"]
    for view in views:
        try:
            load_recognition_model(view)
            logger.info("Pre-loaded recognition model for '%s' view.", view)
        except FileNotFoundError:
            logger.warning(
                "Recognition model for '%s' view not found — "
                "will fail at request time.", view,
            )
