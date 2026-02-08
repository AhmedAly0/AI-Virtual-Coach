"""
Training scripts for pose-based exercise recognition.

Active experiments:
    - experiment_1: Pose MLP (temporal features, multi-run evaluation)

Archived (GEI-based) experiments are in src/scripts/archive/.
"""

from .experiment_1 import train_experiment_1, train_experiment_1_multi_run

__all__ = [
    'train_experiment_1',
    'train_experiment_1_multi_run',
]
