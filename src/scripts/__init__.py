"""
Training experiments for GEI and pose-based exercise recognition.
"""

from .experiment_1 import train_experiment_1, train_one_run
from .experiment_2 import train_experiment_2, train_one_run_progressive
from .experiment_3 import train_experiment_3, train_one_run as train_one_run_v3
from .experiment_7 import (
    train_experiment_7_temporal,
    train_experiment_7_temporal_multi_run,
    train_experiment_7_static,
    train_experiment_7_static_multi_run,
)

__all__ = [
    'train_experiment_1',
    'train_one_run',
    'train_experiment_2',
    'train_one_run_progressive',
    'train_experiment_3',
    'train_one_run_v3',
    # Experiment 7 - LSTM on pose data
    'train_experiment_7_temporal',
    'train_experiment_7_temporal_multi_run',
    'train_experiment_7_static',
    'train_experiment_7_static_multi_run',
]
