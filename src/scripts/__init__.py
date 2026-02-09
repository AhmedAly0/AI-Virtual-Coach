"""
Training scripts for pose-based exercise recognition.

Active experiments:
    - exercise_recognition: Pose MLP (temporal features, multi-run evaluation)

Archived (GEI-based) experiments are in src/scripts/archive/.
"""

from .exercise_recognition import train_exercise_recognition, train_exercise_recognition_multi_run

__all__ = [
    'train_exercise_recognition',
    'train_exercise_recognition_multi_run',
]
