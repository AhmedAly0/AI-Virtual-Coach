"""
Archived GEI-based experiments (legacy).

These experiments used Gait Energy Images (GEIs) as input and have been
superseded by pose-based models. They are preserved here for reference only.

⚠️  Do NOT import from this package for new work. Use the active
    pose-based experiments in src.scripts instead:
    - experiment_1 (Pose MLP)
"""

import warnings

warnings.warn(
    "The src.scripts.archive package contains deprecated GEI-based experiments. "
    "Use pose-based experiment_1 (Pose MLP) instead.",
    DeprecationWarning,
    stacklevel=2,
)
