"""Experiment 5: Small task-specific CNN with subject-wise k-fold CV."""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    to_int,
    split_by_subject_two_way,
    split_by_subjects_three_way,
    build_subject_folds,
)
from src.data.dataset_builder import build_streaming_dataset
from src.models import build_small_gei_cnn, categorical_with_label_smoothing
from src.utils.io_utils import load_config, set_global_seed, setup_results_folder
from src.utils.metrics import macro_f1_score, per_class_f1_scores

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Only add handler if no root handlers exist (notebook provides central logging)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
logger.propagate = True  # Let messages propagate to root handler


def _build_optimizer(initial_lr: float, weight_decay: float, steps_per_epoch: int, max_epochs: int):
    """Create a cosine-decayed AdamW optimizer with graceful fallbacks.

    Args:
        initial_lr (float): Peak learning rate used at the start of each fold.
        weight_decay (float): AdamW weight decay factor for regularization.
        steps_per_epoch (int): Number of gradient steps taken per epoch.
        max_epochs (int): Maximum epochs used to size the cosine schedule.

    Returns:
        tf.keras.optimizers.Optimizer: AdamW (or best-effort substitute) configured for the run.
    """
    decay_steps = max(1, steps_per_epoch * max_epochs)
    schedule = tf.keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps)
    try:
        return tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    except AttributeError:
        # Fallback for older TF versions
        try:
            return tf.keras.optimizers.experimental.AdamW(learning_rate=schedule, weight_decay=weight_decay)
        except AttributeError:
            logger.warning("AdamW not available. Falling back to Adam optimizer.")
            return tf.keras.optimizers.Adam(learning_rate=schedule)


def _build_dataset(
    samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    batch_size: int,
    img_size: int,
    augment_config: Optional[Dict],
    *,
    shuffle: bool,
    augment: bool,
    color_mode: str
):
    """Wrap streaming dataset builder with Experiment 5 defaults.

    Args:
        samples (List[Tuple[str, np.ndarray, str]]): Raw (label, image, subject) tuples.
        label_to_int (Dict[str, int]): Mapping from label string to numeric id.
        batch_size (int): Batch size for the tf.data pipeline.
        img_size (int): Target spatial resolution for resized GEIs.
        augment_config (Optional[Dict]): Augmentation knobs (flip, random erasing, etc.).
        shuffle (bool): Whether to shuffle the dataset window each epoch.
        augment (bool): Whether to apply data augmentation transforms.
        color_mode (str): "grayscale" or "rgb" to control channel count.

    Returns:
        tf.data.Dataset: Batched streaming dataset configured per the provided flags.
    """
    return build_streaming_dataset(
        samples,
        label_to_int,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=None,
        augment_config=augment_config,
        shuffle=shuffle,
        augment=augment,
        color_mode=color_mode,
    )


def _prepare_labels(samples: List[Tuple[str, np.ndarray, str]], label_to_int: Dict[str, int]) -> np.ndarray:
    """Convert input samples into their corresponding integer labels.

    Args:
        samples (List[Tuple[str, np.ndarray, str]]): Sequence of Experiment 5 tuples.
        label_to_int (Dict[str, int]): Lookup from label string to integer id.

    Returns:
        np.ndarray: Dense vector of integer label ids aligned with `samples` order.
    """
    return np.array([label_to_int[sample[0]] for sample in samples], dtype=np.int32)


def _train_one_fold(
    fold_idx: int,
    train_samples: List[Tuple[str, np.ndarray, str]],
    val_samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    num_classes: int,
    config: Dict,
    augment_config: Optional[Dict],
    color_mode: str,
    results_folder: str
) -> Dict:
    """Run one CV fold, persisting metrics, confusion matrix, and weights.

    Args:
        fold_idx (int): Fold identifier (0-indexed) used for logging and folder names.
        train_samples (List[Tuple[str, np.ndarray, str]]): Samples for training split.
        val_samples (List[Tuple[str, np.ndarray, str]]): Samples for validation split.
        label_to_int (Dict[str, int]): Label encoding shared across folds.
        num_classes (int): Total number of exercise classes in the dataset.
        config (Dict): Parsed Experiment 5 configuration dictionary.
        augment_config (Optional[Dict]): Augmentation parameters dictionary.
        color_mode (str): "grayscale" or "rgb" to match model inputs.
        results_folder (str): Base folder where fold artefacts should be stored.

    Returns:
        Dict: Serialized fold metrics (loss/accuracy/F1/confusion) ready for aggregation.
    """
    if not train_samples:
        raise ValueError(
            f"Fold {fold_idx}: training split is empty after subject-wise assignment. "
            "Check cross-validation configuration and subject distribution."
        )

    if not val_samples:
        raise ValueError(
            f"Fold {fold_idx}: validation split is empty after subject-wise assignment. "
            "Check cross-validation configuration and subject distribution."
        )

    img_size = config['model']['img_size']
    batch_size = config['training']['batch_size']
    max_epochs = config['training']['max_epochs']
    weight_decay = config['training']['weight_decay']
    initial_lr = config['training']['initial_lr']
    label_smoothing = config['training']['label_smoothing']

    logger.info(
        "Fold %s: entering _train_one_fold with %s train samples, %s val samples",
        fold_idx,
        len(train_samples),
        len(val_samples),
    )

    train_ds = _build_dataset(
        train_samples,
        label_to_int,
        batch_size,
        img_size,
        augment_config,
        shuffle=True,
        augment=True,
        color_mode=color_mode,
    )
    val_ds = _build_dataset(
        val_samples,
        label_to_int,
        batch_size,
        img_size,
        augment_config,
        shuffle=False,
        augment=False,
        color_mode=color_mode,
    )

    steps_per_epoch = max(1, math.ceil(len(train_samples) / batch_size))

    model = build_small_gei_cnn(
        img_size=img_size,
        num_classes=num_classes,
        dense_units=config['model'].get('dense_units', 128),
        input_channels=config['model'].get('input_channels', 1),
        dropout_rate=0.35,
    )

    optimizer = _build_optimizer(initial_lr, weight_decay, steps_per_epoch, max_epochs)
    loss_fn = categorical_with_label_smoothing(num_classes, label_smoothing)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    callbacks: List[tf.keras.callbacks.Callback] = []
    early_stopping_callback = None
    if early_cfg.get('enabled', True):
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=early_cfg.get('monitor', 'val_loss'),
            patience=early_cfg.get('patience', 10),
            min_delta=early_cfg.get('min_delta', 1e-4),
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping_callback)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Calculate best epoch correctly
    if early_stopping_callback is not None and early_stopping_callback.stopped_epoch > 0:
        # Early stopping triggered
        best_epoch = early_stopping_callback.stopped_epoch - early_stopping_callback.patience + 1
    else:
        # Trained to max_epochs without early stopping, find epoch with best val_loss
        best_epoch = int(np.argmin(history.history['val_loss'])) + 1

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    val_targets = _prepare_labels(val_samples, label_to_int)
    val_preds = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    val_macro_f1 = macro_f1_score(val_targets, val_preds, num_classes)
    per_class_f1 = per_class_f1_scores(val_targets, val_preds, num_classes)
    confusion = tf.math.confusion_matrix(val_targets, val_preds, num_classes=num_classes).numpy().tolist()

    fold_dir = os.path.join(results_folder, 'folds', f'fold_{fold_idx:02d}')
    os.makedirs(fold_dir, exist_ok=True)

    fold_metrics = {
        'fold_idx': fold_idx,
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'val_macro_f1': float(val_macro_f1),
        'per_class_f1': per_class_f1,
        'history': history.history,
        'best_epoch': int(best_epoch),
        'confusion_matrix': confusion,
    }

    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    model.save_weights(os.path.join(fold_dir, 'model.weights.h5'))

    return fold_metrics, model


def _train_final_model(
    train_samples: List[Tuple[str, np.ndarray, str]],
    val_samples: List[Tuple[str, np.ndarray, str]],
    test_samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    num_classes: int,
    config: Dict,
    augment_config: Optional[Dict],
    color_mode: str,
    results_folder: str
) -> Dict:
    """Train on the pooled subjects, evaluate on the frozen test set, and store outputs.

    Args:
        train_samples (List[Tuple[str, np.ndarray, str]]): Samples used for final training.
        val_samples (List[Tuple[str, np.ndarray, str]]): Optional validation split for early stopping.
        test_samples (List[Tuple[str, np.ndarray, str]]): Held-out 30% subject set.
        label_to_int (Dict[str, int]): Shared label encoding dictionary.
        num_classes (int): Number of exercise labels.
        config (Dict): Experiment 5 configuration.
        augment_config (Optional[Dict]): Augmentation parameters.
        color_mode (str): Input color mode string for dataset builders.
        results_folder (str): Run directory where final metrics/weights are saved.

    Returns:
        Dict: Final evaluation metrics, confusion matrix, and training history.
    """
    if not train_samples:
        raise ValueError("Final training split is empty. Cannot fit the model.")

    if not val_samples:
        raise ValueError("Final validation split is empty. Provide at least one validation subject.")

    if not test_samples:
        raise ValueError("Final test split is empty. Hold out at least one subject for testing.")

    img_size = config['model']['img_size']
    batch_size = config['training']['batch_size']
    max_epochs = config['training']['max_epochs']
    weight_decay = config['training']['weight_decay']
    initial_lr = config['training']['initial_lr']
    label_smoothing = config['training']['label_smoothing']

    train_ds = _build_dataset(
        train_samples,
        label_to_int,
        batch_size,
        img_size,
        augment_config,
        shuffle=True,
        augment=True,
        color_mode=color_mode,
    )
    val_ds = _build_dataset(
        val_samples,
        label_to_int,
        batch_size,
        img_size,
        augment_config,
        shuffle=False,
        augment=False,
        color_mode=color_mode,
    )
    test_ds = _build_dataset(
        test_samples,
        label_to_int,
        batch_size,
        img_size,
        augment_config,
        shuffle=False,
        augment=False,
        color_mode=color_mode,
    )

    steps_per_epoch = max(1, math.ceil(len(train_samples) / batch_size))
    model = build_small_gei_cnn(
        img_size=img_size,
        num_classes=num_classes,
        dense_units=config['model'].get('dense_units', 128),
        input_channels=config['model'].get('input_channels', 1),
        dropout_rate=0.35,
    )

    optimizer = _build_optimizer(initial_lr, weight_decay, steps_per_epoch, max_epochs)
    loss_fn = categorical_with_label_smoothing(num_classes, label_smoothing)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    callbacks: List[tf.keras.callbacks.Callback] = []
    if early_cfg.get('enabled', True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get('monitor', 'val_loss'),
                patience=early_cfg.get('patience', 10),
                min_delta=early_cfg.get('min_delta', 1e-4),
                restore_best_weights=True,
                verbose=1,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    test_targets = _prepare_labels(test_samples, label_to_int)
    test_preds = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    test_macro_f1 = macro_f1_score(test_targets, test_preds, num_classes)
    per_class_f1 = per_class_f1_scores(test_targets, test_preds, num_classes)
    confusion = tf.math.confusion_matrix(test_targets, test_preds, num_classes=num_classes).numpy().tolist()

    final_dir = os.path.join(results_folder, 'final_model')
    os.makedirs(final_dir, exist_ok=True)

    final_metrics = {
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_macro_f1),
        'per_class_f1': per_class_f1,
        'confusion_matrix': confusion,
        'history': history.history,
    }

    with open(os.path.join(final_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    model.save_weights(os.path.join(final_dir, 'model.weights.h5'))

    return final_metrics


def train_experiment_5(
    dataset: List[Tuple[str, np.ndarray, str]],
    config_path: str = 'config/experiment_5.yaml'
) -> Dict:
    """Execute the full Experiment 5 pipeline (CV + retrain) and report artefacts.

    Args:
        dataset (List[Tuple[str, np.ndarray, str]]): Subject-tagged GEI samples.
        config_path (str): Path to the Experiment 5 YAML configuration file.

    Returns:
        Dict: Bundle containing run index, per-fold metrics, final metrics, and output paths.
    """

    if not dataset:
        raise ValueError('Dataset cannot be empty')

    config = load_config(config_path)
    set_global_seed(config.get('dataset', {}).get('random_seed', 42))
    augment_config = config.get('augmentation', {})
    color_mode = 'grayscale' if config['model'].get('input_channels', 1) == 1 else 'rgb'

    base_results_dir = config['results']['base_dir']
    results_folder, run_idx = setup_results_folder(base_results_dir)
    logger.info("Experiment 5 run directory: %s", results_folder)

    label_to_int = to_int([sample[0] for sample in dataset])
    num_classes = len(label_to_int)

    pool_samples, test_samples = split_by_subject_two_way(
        dataset,
        split_ratio=config['dataset']['test_ratio'],
        seed=config['dataset'].get('random_seed', 42),
        stratified=True  # Ensure all 15 exercises in held-out test set
    )

    cv_config = config['dataset']['cv']
    folds = build_subject_folds(
        pool_samples,
        num_folds=cv_config.get('num_folds', 5),
        seed=config['dataset'].get('random_seed', 42),
        stratified=cv_config.get('stratified_subjects', True)
    )

    for idx, fold in enumerate(folds, start=1):
        unique_subjects = len({sample[2] for sample in fold})
        logger.info(
            "Fold %s diagnostic: %s samples, %s unique subjects",
            idx,
            len(fold),
            unique_subjects,
        )

    fold_results = []
    for fold_position, val_samples in enumerate(folds):
        fold_idx = fold_position + 1
        train_samples = [
            item
            for idx, fold in enumerate(folds)
            if idx != fold_position
            for item in fold
        ]
        logger.info(
            "Fold %s: assembled %s train / %s val samples before training",
            fold_idx,
            len(train_samples),
            len(val_samples),
        )
        fold_metrics, _ = _train_one_fold(
            fold_idx,
            train_samples,
            val_samples,
            label_to_int,
            num_classes,
            config,
            augment_config,
            color_mode,
            results_folder,
        )
        fold_results.append(fold_metrics)

        if config['memory'].get('clear_session_after_run', True):
            tf.keras.backend.clear_session()
        if config['memory'].get('force_gc_after_run', True):
            import gc
            gc.collect()

    cv_summary_path = os.path.join(results_folder, 'cv_summary.json')
    with open(cv_summary_path, 'w') as f:
        json.dump(fold_results, f, indent=2)

    # Retrain on full 70% pool using optional val split
    retrain_cfg = config.get('retrain', {})
    if retrain_cfg.get('use_internal_val_split', True):
        train_final, val_final = split_by_subject_two_way(
            pool_samples,
            split_ratio=retrain_cfg.get('val_ratio', 0.15),
            seed=config['dataset'].get('random_seed', 42),
            stratified=True  # Ensure all exercises in both train and val for final model
        )
    else:
        train_final, val_final = pool_samples, []

    final_metrics = _train_final_model(
        train_final,
        val_final or pool_samples,
        test_samples,
        label_to_int,
        num_classes,
        config,
        augment_config,
        color_mode,
        results_folder,
    )

    if config['memory'].get('clear_session_after_run', True):
        tf.keras.backend.clear_session()
    if config['memory'].get('force_gc_after_run', True):
        import gc
        gc.collect()

    result_bundle = {
        'run_idx': run_idx,
        'cv_results': fold_results,
        'final_metrics': final_metrics,
        'results_folder': results_folder,
    }

    with open(os.path.join(results_folder, 'summary.json'), 'w') as f:
        json.dump(result_bundle, f, indent=2)

    logger.info("Experiment 5 completed. Test macro-F1: %.4f", final_metrics['test_macro_f1'])
    return result_bundle


def _compute_aggregation_stats(all_run_results: List[Dict]) -> Dict:
    """Calculate mean/std/min/max statistics across multiple training runs.

    Args:
        all_run_results (List[Dict]): List of metrics dictionaries from individual runs.

    Returns:
        Dict: Aggregated statistics for test_accuracy, test_macro_f1, and per_class_f1.
    """
    test_accuracies = [run['test_accuracy'] for run in all_run_results]
    test_f1s = [run['test_macro_f1'] for run in all_run_results]
    
    stats = {
        'test_accuracy': {
            'mean': float(np.mean(test_accuracies)),
            'std': float(np.std(test_accuracies)),
            'min': float(np.min(test_accuracies)),
            'max': float(np.max(test_accuracies)),
        },
        'test_macro_f1': {
            'mean': float(np.mean(test_f1s)),
            'std': float(np.std(test_f1s)),
            'min': float(np.min(test_f1s)),
            'max': float(np.max(test_f1s)),
        }
    }
    
    # Aggregate per-class F1 scores
    if all_run_results and 'per_class_f1' in all_run_results[0]:
        num_classes = len(all_run_results[0]['per_class_f1'])
        per_class_f1_aggregated = {}
        
        for class_idx in range(num_classes):
            class_f1s = [run['per_class_f1'][str(class_idx)] for run in all_run_results]
            per_class_f1_aggregated[str(class_idx)] = {
                'mean': float(np.mean(class_f1s)),
                'std': float(np.std(class_f1s)),
                'min': float(np.min(class_f1s)),
                'max': float(np.max(class_f1s)),
            }
        
        stats['per_class_f1'] = per_class_f1_aggregated
    
    return stats


def _save_multi_run_summary(
    all_run_results: List[Dict],
    aggregated_stats: Dict,
    multi_run_folder: str
):
    """Persist multi-run results and aggregated statistics to disk.

    Args:
        all_run_results (List[Dict]): Complete metrics for all individual runs.
        aggregated_stats (Dict): Mean/std/min/max statistics across runs.
        multi_run_folder (str): Base folder for multi-run experiment.
    """
    os.makedirs(multi_run_folder, exist_ok=True)
    
    # Save aggregated statistics (JSON)
    with open(os.path.join(multi_run_folder, 'aggregated_stats.json'), 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    # Save aggregated statistics (human-readable text)
    with open(os.path.join(multi_run_folder, 'aggregated_stats.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT 5: 30-RUN MULTI-SEED AGGREGATED STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Accuracy:\n")
        f.write(f"  Mean: {aggregated_stats['test_accuracy']['mean']:.4f}\n")
        f.write(f"  Std:  {aggregated_stats['test_accuracy']['std']:.4f}\n")
        f.write(f"  Min:  {aggregated_stats['test_accuracy']['min']:.4f}\n")
        f.write(f"  Max:  {aggregated_stats['test_accuracy']['max']:.4f}\n\n")
        
        f.write("Test Macro F1:\n")
        f.write(f"  Mean: {aggregated_stats['test_macro_f1']['mean']:.4f}\n")
        f.write(f"  Std:  {aggregated_stats['test_macro_f1']['std']:.4f}\n")
        f.write(f"  Min:  {aggregated_stats['test_macro_f1']['min']:.4f}\n")
        f.write(f"  Max:  {aggregated_stats['test_macro_f1']['max']:.4f}\n\n")
        
        if 'per_class_f1' in aggregated_stats:
            f.write("Per-Class F1 (Mean ± Std):\n")
            for class_idx, class_stats in sorted(aggregated_stats['per_class_f1'].items()):
                f.write(f"  Class {class_idx}: {class_stats['mean']:.4f} ± {class_stats['std']:.4f}\n")
    
    # Save detailed run results
    with open(os.path.join(multi_run_folder, 'runs_detail.json'), 'w') as f:
        json.dump(all_run_results, f, indent=2)
    
    logger.info("Multi-run summary saved to %s", multi_run_folder)


def train_experiment_5_multi_run(
    dataset: List[Tuple[str, np.ndarray, str]],
    config_path: str = 'config/experiment_5.yaml'
) -> Tuple[List[Dict], Dict]:
    """Execute 30 independent training runs with different random seeds and aggregate results.

    Args:
        dataset (List[Tuple[str, np.ndarray, str]]): Subject-tagged GEI samples.
        config_path (str): Path to the Experiment 5 YAML configuration file.

    Returns:
        Tuple[List[Dict], Dict]: (all_run_results, aggregated_stats)
            - all_run_results: List of metrics dictionaries from each run
            - aggregated_stats: Mean/std/min/max for test_accuracy and test_macro_f1
    """
    if not dataset:
        raise ValueError('Dataset cannot be empty')
    
    config = load_config(config_path)
    multi_run_config = config.get('multi_run', {})
    
    if not multi_run_config.get('enabled', False):
        raise ValueError(
            "multi_run.enabled must be set to true in config to run this function. "
            "Set it in config/experiment_5.yaml or use train_experiment_5() for CV mode."
        )
    
    num_runs = multi_run_config.get('num_runs', 30)
    base_seed = multi_run_config.get('base_seed', 42)
    save_all_runs = multi_run_config.get('save_all_runs', True)
    save_all_cm = multi_run_config.get('save_all_confusion_matrices', True)
    
    augment_config = config.get('augmentation', {})
    color_mode = 'grayscale' if config['model'].get('input_channels', 1) == 1 else 'rgb'
    
    # Create multi-run parent directory
    base_results_dir = config['results']['base_dir']
    multi_run_parent = os.path.join(base_results_dir, 'multi_run')
    
    # Find next multi_run index
    run_idx = 1
    while os.path.exists(os.path.join(multi_run_parent, f'multi_run_{run_idx:03d}')):
        run_idx += 1
    
    multi_run_folder = os.path.join(multi_run_parent, f'multi_run_{run_idx:03d}')
    os.makedirs(multi_run_folder, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting Experiment 5 Multi-Run (30 seeds)")
    logger.info("Multi-run folder: %s", multi_run_folder)
    logger.info("=" * 60)
    
    # Get label encoding
    label_to_int = to_int([sample[0] for sample in dataset])
    num_classes = len(label_to_int)
    
    all_run_results = []
    
    for run_number in range(1, num_runs + 1):
        run_seed = base_seed + run_number
        logger.info("\n" + "=" * 60)
        logger.info("Run %d/%d (seed=%d)", run_number, num_runs, run_seed)
        logger.info("=" * 60)
        
        # Override config seed for this run
        set_global_seed(run_seed)
        
        # Create 3-way stratified split with current seed
        train_samples, val_samples, test_samples = split_by_subjects_three_way(
            dataset,
            val_ratio=config.get('retrain', {}).get('val_ratio', 0.15),
            test_ratio=config['dataset']['test_ratio'],
            seed=run_seed,
            stratified=True
        )
        
        logger.info(
            "Split sizes: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples)
        )
        
        # Create run-specific folder
        run_folder = os.path.join(multi_run_folder, f'run_{run_number:03d}')
        os.makedirs(run_folder, exist_ok=True)
        
        # Train final model for this run (no CV)
        final_metrics = _train_final_model(
            train_samples,
            val_samples,
            test_samples,
            label_to_int,
            num_classes,
            config,
            augment_config,
            color_mode,
            run_folder
        )
        
        # Add run metadata
        run_result = {
            'run_idx': run_number,
            'seed': run_seed,
            **final_metrics
        }
        
        all_run_results.append(run_result)
        
        logger.info(
            "Run %d completed: test_acc=%.4f, test_f1=%.4f",
            run_number,
            final_metrics['test_accuracy'],
            final_metrics['test_macro_f1']
        )
        
        # Memory cleanup
        if config['memory'].get('clear_session_after_run', True):
            tf.keras.backend.clear_session()
        if config['memory'].get('force_gc_after_run', True):
            import gc
            gc.collect()
    
    # Compute aggregated statistics
    aggregated_stats = _compute_aggregation_stats(all_run_results)
    
    # Save all results
    _save_multi_run_summary(all_run_results, aggregated_stats, multi_run_folder)
    
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-RUN COMPLETE")
    logger.info("=" * 60)
    logger.info("Test Accuracy:  %.4f ± %.4f", aggregated_stats['test_accuracy']['mean'], aggregated_stats['test_accuracy']['std'])
    logger.info("Test Macro F1:  %.4f ± %.4f", aggregated_stats['test_macro_f1']['mean'], aggregated_stats['test_macro_f1']['std'])
    logger.info("Results saved to: %s", multi_run_folder)
    logger.info("=" * 60)
    
    return all_run_results, aggregated_stats


__all__ = ['train_experiment_5', 'train_experiment_5_multi_run']


if __name__ == '__main__':
    logger.warning(
        "Experiment 5 expects a pre-loaded dataset. Import this module and call\n"
        "train_experiment_5(dataset=...) after loading your subject-annotated GEIs."
    )
