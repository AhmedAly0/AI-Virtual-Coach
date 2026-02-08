"""Grid search utilities for BN+Dropout+Weight Decay exercise recognition model.

This module is added because repeated in-place edits to `rgb_gei_lib_v1.py` did not persist the
intended grid search helper functions. Import from here instead of relying on them being
embedded inside the large original library file.

Usage (in notebook):
    from gei_grid_search_utils import grid_search_bn_dropout, summarize_runs
    results_list, best = grid_search_bn_dropout(dataset, base_configs)

Dependencies: expects the following symbols to exist in `rgb_gei_lib_v1`:
    - create_model_for_exer_recog_bn_dropout
    - complete_experiment_with_model_factory
    - train_test_split_by_subject (or whichever split function you pass)

All confusion matrices are padded/truncated to 15x15 to keep shape consistency.
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, List, Dict, Any, Sequence, Optional

# Import required factories / runners from the original library
from rgb_gei_lib_v1 import (
    create_model_for_exer_recog_bn_dropout,
    complete_experiment_with_model_factory,
    train_test_split_by_subject,  # default split
)

NUM_CLASSES = 15

__all__ = [
    "_compute_per_class_recall",
    "summarize_runs",
    "grid_search_bn_dropout",
]

def _pad_confusion_matrix(cm: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.asarray(cm)
    if cm.shape == (num_classes, num_classes):
        return cm
    padded = np.zeros((num_classes, num_classes), dtype=cm.dtype)
    r = min(num_classes, cm.shape[0])
    c = min(num_classes, cm.shape[1])
    padded[:r, :c] = cm[:r, :c]
    return padded

def _compute_per_class_recall(confusion_matrix: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if confusion_matrix is None:
        return None, None
    cm = _pad_confusion_matrix(confusion_matrix, NUM_CLASSES)
    with np.errstate(divide='ignore', invalid='ignore'):
        support = cm.sum(axis=1)
        correct = np.diag(cm)
        recalls = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support!=0)
    macro_recall = float(recalls.mean()) if recalls.size else 0.0
    return recalls, macro_recall

def summarize_runs(results: Dict[str, Any], num_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    accs = np.array(results.get('accuracies', []), dtype=float).flatten()
    cms = results.get('confusion_matrices', [])
    macro_vals = []
    for cm in cms:
        _, mr = _compute_per_class_recall(cm)
        macro_vals.append(mr)
    macro_arr = np.array(macro_vals) if macro_vals else np.array([])
    best_idx = int(accs.argmax()) if accs.size else 0
    best_cm = _pad_confusion_matrix(cms[best_idx]) if cms else None
    return {
        'accuracy_mean': float(accs.mean()) if accs.size else 0.0,
        'accuracy_std': float(accs.std(ddof=1)) if accs.size > 1 else 0.0,
        'macro_recall_mean': float(macro_arr.mean()) if macro_arr.size else 0.0,
        'macro_recall_std': float(macro_arr.std(ddof=1)) if macro_arr.size > 1 else 0.0,
        'best_confusion_matrix': best_cm,
        'per_run_macro_recall': macro_arr,
        'per_run_accuracies': accs,
    }


