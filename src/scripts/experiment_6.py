"""Experiment 6: Pose-based MLP baseline for exercise recognition using static features."""

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_pose_data
from src.data.data_loader import load_pose_temporal_data
from src.data.dataset_builder import make_pose_split_three_way, build_pose_datasets_three_way
from src.utils.io_utils import set_global_seed, load_config, setup_results_folder
from src.utils.metrics import macro_f1_score, per_class_f1_scores

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
logger.propagate = True


def _filter_pose_features(dataset: List[Tuple], selected_angles: List[str], all_angle_names: List[str]) -> List[Tuple]:
    """Filter pose features to include only selected angles.
    
    Args:
        dataset (List[Tuple]): Dataset with (label, features, subject_id, view) tuples.
        selected_angles (List[str]): List of angle names to keep (e.g., ['left_elbow', 'right_knee']).
        all_angle_names (List[str]): Complete list of angle names from NPZ file.
    
    Returns:
        List[Tuple]: Filtered dataset with reduced feature dimensionality.
    """
    if not selected_angles or selected_angles == ['all']:
        return dataset  # Use all angles
    
    # Find indices of selected angles
    angle_indices = []
    for angle_name in selected_angles:
        if angle_name in all_angle_names:
            angle_indices.append(all_angle_names.index(angle_name))
        else:
            logger.warning(f"Angle '{angle_name}' not found in dataset. Available: {all_angle_names}")
    
    if not angle_indices:
        raise ValueError(f"No valid angles selected. Available angles: {all_angle_names}")
    
    # Each angle has 5 features (mean, std, min, max, range)
    # For static features: columns are organized as [angle0_feat0-4, angle1_feat0-4, ...]
    feature_indices = []
    for angle_idx in angle_indices:
        start_col = angle_idx * 5
        feature_indices.extend(range(start_col, start_col + 5))
    
    # Filter features for each sample
    filtered_dataset = []
    for item in dataset:
        # Handle both 3-tuple (label, features, subject_id) and 4-tuple (label, features, subject_id, view)
        if len(item) == 4:
            label, features, subject_id, view = item
            filtered_features = features[feature_indices]
            filtered_dataset.append((label, filtered_features, subject_id, view))
        else:
            label, features, subject_id = item
            filtered_features = features[feature_indices]
            filtered_dataset.append((label, filtered_features, subject_id))
    
    logger.info(
        "Filtered features: %d angles selected (%s) -> %d features (from original %d)",
        len(angle_indices),
        ', '.join([all_angle_names[i] for i in angle_indices]),
        len(feature_indices),
        features.shape[0]
    )
    
    return filtered_dataset


def _build_mlp(input_dim: int, num_classes: int, hidden_sizes: Tuple[int, ...], dropout: float, lr: float) -> tf.keras.Model:
    """Simple MLP baseline for pose static features.

    Args:
        input_dim (int): Feature dimension of pose vectors (45 for single view).
        num_classes (int): Number of exercise classes.
        hidden_sizes (Tuple[int, ...]): Dense layer sizes.
        dropout (float): Dropout rate applied after each hidden layer.
        lr (float): Learning rate for Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in hidden_sizes:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def _build_callbacks(config: Dict) -> List[tf.keras.callbacks.Callback]:
    """Build training callbacks from configuration.

    Args:
        config (Dict): Full configuration dictionary with callbacks section.

    Returns:
        List[tf.keras.callbacks.Callback]: List of configured callbacks.
    """
    callbacks: List[tf.keras.callbacks.Callback] = []

    # Early stopping callback
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    if early_cfg.get('enabled', True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get('monitor', 'val_loss'),
                patience=early_cfg.get('patience', 10),
                restore_best_weights=early_cfg.get('restore_best_weights', True),
                min_delta=early_cfg.get('min_delta', 0.0),
                verbose=1,
            )
        )

    # ReduceLROnPlateau callback
    reduce_lr_cfg = config.get('callbacks', {}).get('reduce_lr_on_plateau', {})
    if reduce_lr_cfg.get('enabled', False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_cfg.get('monitor', 'val_loss'),
                factor=reduce_lr_cfg.get('factor', 0.5),
                patience=reduce_lr_cfg.get('patience', 15),
                min_lr=reduce_lr_cfg.get('min_lr', 1e-7),
                verbose=reduce_lr_cfg.get('verbose', 1),
            )
        )

    return callbacks


def train_experiment_6_static(
    npz_path: str,
    config_path: str = 'config/experiment_6.yaml',
    *,
    results_folder: Optional[str] = None,
    run_idx: Optional[int] = None,
    # Legacy parameters for backward compatibility
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    hidden_sizes: Optional[Tuple[int, ...]] = None,
    dropout: Optional[float] = None,
    lr: Optional[float] = None,
    stratified: Optional[bool] = None,
    max_epochs: Optional[int] = None,
) -> Dict:
    """Train a pose-based MLP baseline with subject-wise stratified splits using static features.

    Args:
        npz_path (str): Path to pose static NPZ file (front or side view).
        config_path (str): Path to YAML configuration file. Defaults to 'config/experiment_6.yaml'.
        results_folder (Optional[str]): Pre-created results folder path (used in multi-run mode).
        run_idx (Optional[int]): Run index for naming (used in multi-run mode).
        
        Legacy parameters (overridden by config if None):
        val_ratio (Optional[float]): Fraction of subjects for validation split.
        test_ratio (Optional[float]): Fraction of subjects for test split.
        seed (Optional[int]): Random seed for reproducibility.
        batch_size (Optional[int]): Batch size for tf.data pipelines.
        hidden_sizes (Optional[Tuple[int, ...]]): Hidden layer widths for the MLP.
        dropout (Optional[float]): Dropout rate after each hidden layer.
        lr (Optional[float]): Learning rate for Adam.
        stratified (Optional[bool]): Ensure all classes appear in train/val/test when possible.
        max_epochs (Optional[int]): Training epochs with early stopping.

    Returns:
        Dict: Training history, label maps, split sizes, and test metrics including macro F1.
    """

    # Load configuration
    config = load_config(config_path)
    
    # Extract hyperparameters from config (or use legacy parameters if provided)
    val_ratio = val_ratio if val_ratio is not None else config['dataset']['val_ratio']
    test_ratio = test_ratio if test_ratio is not None else config['dataset']['test_ratio']
    seed = seed if seed is not None else config['dataset']['random_seed']
    batch_size = batch_size if batch_size is not None else config['training']['batch_size']
    hidden_sizes = hidden_sizes if hidden_sizes is not None else tuple(config['model']['hidden_sizes'])
    dropout = dropout if dropout is not None else config['model']['dropout']
    lr = lr if lr is not None else config['training']['lr']
    stratified = stratified if stratified is not None else config['dataset']['stratified']
    max_epochs = max_epochs if max_epochs is not None else config['training']['max_epochs']
    
    # Setup results folder if not provided (single-run mode)
    if results_folder is None:
        base_results_dir = config['results']['base_dir']
        results_folder, run_idx = setup_results_folder(base_results_dir)
        logger.info("Experiment 6 (static) results directory: %s", results_folder)
    
    set_global_seed(seed)

    dataset, pose_summary = load_pose_data(npz_path)
    
    # Filter features based on selected angles (if specified in config)
    selected_angles = config.get('dataset', {}).get('selected_angles', ['all'])
    if selected_angles and selected_angles != ['all']:
        if 'angle_names' not in pose_summary:
            raise KeyError(
                "NPZ file is missing 'angle_names' field but config specifies selected_angles. "
                "Please regenerate the NPZ file or set selected_angles to ['all']."
            )
        angle_names = pose_summary['angle_names']
        dataset = _filter_pose_features(dataset, selected_angles, angle_names)

    train_samples, val_samples, test_samples, label_to_int = make_pose_split_three_way(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratified=stratified,
    )

    train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test = build_pose_datasets_three_way(
        train_samples,
        val_samples,
        test_samples,
        label_to_int,
        batch_size=batch_size,
    )

    input_dim = X_train.shape[1]
    num_classes = len(label_to_int)
    int_to_label = {v: k for k, v in label_to_int.items()}

    model = _build_mlp(input_dim, num_classes, hidden_sizes, dropout, lr)

    # Build callbacks from config
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    callbacks: List[tf.keras.callbacks.Callback] = []
    if early_cfg.get('enabled', True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get('monitor', 'val_loss'),
                patience=early_cfg.get('patience', 10),
                restore_best_weights=early_cfg.get('restore_best_weights', True),
                min_delta=early_cfg.get('min_delta', 0.0),
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

    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)

    macro_f1 = macro_f1_score(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    per_class_f1 = per_class_f1_scores(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    # Extract hyperparameters summary for saving
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    reduce_lr_cfg = config.get('callbacks', {}).get('reduce_lr_on_plateau', {})
    hyperparameters = {
        'batch_size': batch_size,
        'hidden_sizes': list(hidden_sizes),
        'dropout': dropout,
        'learning_rate': lr,
        'max_epochs': max_epochs,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'stratified': stratified,
        'selected_angles': config.get('dataset', {}).get('selected_angles', ['all']),
        'early_stopping': {
            'enabled': early_cfg.get('enabled', True),
            'monitor': early_cfg.get('monitor', 'val_loss'),
            'patience': early_cfg.get('patience', 10),
            'min_delta': early_cfg.get('min_delta', 0.0),
        },
        'reduce_lr_on_plateau': {
            'enabled': reduce_lr_cfg.get('enabled', False),
            'monitor': reduce_lr_cfg.get('monitor', 'val_loss'),
            'factor': reduce_lr_cfg.get('factor', 0.5),
            'patience': reduce_lr_cfg.get('patience', 15),
            'min_lr': reduce_lr_cfg.get('min_lr', 1e-7),
        },
        'input_dim': input_dim,
        'num_classes': num_classes,
    }

    results = {
        'run_idx': run_idx,
        'seed': seed,
        'hyperparameters': hyperparameters,
        'pose_summary': pose_summary,
        'label_to_int': label_to_int,
        'int_to_label': int_to_label,
        'history': history.history,
        'test_metrics': {
            'loss': float(test_eval['loss']),
            'accuracy': float(test_eval.get('accuracy', 0.0)),
            'macro_f1': macro_f1,
            'per_class_f1': {int(k): float(v) for k, v in per_class_f1.items()},
            'confusion_matrix': conf_matrix.tolist(),
        },
        'train_sizes': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
        },
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
    }

    # Save results to disk
    if config['results'].get('save_metrics', True):
        metrics_path = os.path.join(results_folder, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(
        "Experiment 6 (static) complete. Test acc=%.4f, macro F1=%.4f",
        results['test_metrics']['accuracy'],
        results['test_metrics']['macro_f1'],
    )

    return results


def _compute_aggregation_stats_exp6(all_run_results: List[Dict]) -> Dict:
    """Calculate mean/std/min/max statistics across multiple training runs.

    Args:
        all_run_results (List[Dict]): List of results dictionaries from individual runs.

    Returns:
        Dict: Aggregated statistics for test_accuracy, test_macro_f1, and per_class_f1.
    """
    test_accuracies = [run['test_metrics']['accuracy'] for run in all_run_results]
    test_macro_f1s = [run['test_metrics']['macro_f1'] for run in all_run_results]

    # Aggregate per-class F1 scores
    num_classes = len(all_run_results[0]['test_metrics']['per_class_f1'])
    per_class_f1_all_runs = {
        class_idx: [run['test_metrics']['per_class_f1'][class_idx] for run in all_run_results]
        for class_idx in range(num_classes)
    }

    aggregated_stats = {
        'test_accuracy': {
            'mean': float(np.mean(test_accuracies)),
            'std': float(np.std(test_accuracies)),
            'min': float(np.min(test_accuracies)),
            'max': float(np.max(test_accuracies)),
            'all_values': test_accuracies,
        },
        'test_macro_f1': {
            'mean': float(np.mean(test_macro_f1s)),
            'std': float(np.std(test_macro_f1s)),
            'min': float(np.min(test_macro_f1s)),
            'max': float(np.max(test_macro_f1s)),
            'all_values': test_macro_f1s,
        },
        'per_class_f1': {
            int(class_idx): {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
            }
            for class_idx, f1_scores in per_class_f1_all_runs.items()
        },
        'num_runs': len(all_run_results),
    }

    return aggregated_stats


def _save_multi_run_summary_exp6(
    all_run_results: List[Dict],
    aggregated_stats: Dict,
    multi_run_folder: str
):
    """Save aggregated statistics and individual run results to disk.

    Args:
        all_run_results (List[Dict]): List of all individual run results.
        aggregated_stats (Dict): Pre-computed aggregation statistics.
        multi_run_folder (str): Directory path for saving multi-run outputs.
    """
    # Save aggregated statistics as JSON
    stats_path = os.path.join(multi_run_folder, 'aggregated_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)

    # Save human-readable summary
    summary_path = os.path.join(multi_run_folder, 'aggregated_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT 6 MULTI-RUN SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of runs: {aggregated_stats['num_runs']}\n\n")
        
        f.write("TEST ACCURACY:\n")
        f.write(f"  Mean: {aggregated_stats['test_accuracy']['mean']:.4f}\n")
        f.write(f"  Std:  {aggregated_stats['test_accuracy']['std']:.4f}\n")
        f.write(f"  Min:  {aggregated_stats['test_accuracy']['min']:.4f}\n")
        f.write(f"  Max:  {aggregated_stats['test_accuracy']['max']:.4f}\n\n")
        
        f.write("TEST MACRO F1:\n")
        f.write(f"  Mean: {aggregated_stats['test_macro_f1']['mean']:.4f}\n")
        f.write(f"  Std:  {aggregated_stats['test_macro_f1']['std']:.4f}\n")
        f.write(f"  Min:  {aggregated_stats['test_macro_f1']['min']:.4f}\n")
        f.write(f"  Max:  {aggregated_stats['test_macro_f1']['max']:.4f}\n\n")
        
        f.write("PER-CLASS F1 SCORES (mean ± std):\n")
        int_to_label = all_run_results[0]['int_to_label']
        for class_idx in sorted(aggregated_stats['per_class_f1'].keys()):
            class_name = int_to_label[class_idx]
            mean_f1 = aggregated_stats['per_class_f1'][class_idx]['mean']
            std_f1 = aggregated_stats['per_class_f1'][class_idx]['std']
            f.write(f"  {class_name}: {mean_f1:.4f} ± {std_f1:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    # Save all individual run results
    all_runs_path = os.path.join(multi_run_folder, 'all_runs.json')
    with open(all_runs_path, 'w') as f:
        json.dump(all_run_results, f, indent=2)


def train_experiment_6_multi_run(
    npz_path: str,
    config_path: str = 'config/experiment_6.yaml'
) -> Tuple[List[Dict], Dict]:
    """Execute multiple training runs with different seeds and aggregate results.

    Args:
        npz_path (str): Path to pose static NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.

    Returns:
        Tuple[List[Dict], Dict]: (all_run_results, aggregated_stats) containing individual
            run metrics and aggregated statistics across all runs.
    """
    config = load_config(config_path)
    
    # Validate multi-run is enabled
    multi_run_cfg = config.get('multi_run', {})
    if not multi_run_cfg.get('enabled', False):
        raise ValueError(
            "Multi-run mode is disabled in config. Set 'multi_run.enabled: true' "
            "or use train_experiment_6_static() for single runs."
        )
    
    num_runs = multi_run_cfg.get('num_runs', 30)
    base_seed = multi_run_cfg.get('base_seed', 42)
    save_all_runs = multi_run_cfg.get('save_all_runs', True)
    
    # Create multi-run folder inside base_dir
    base_results_dir = config['results']['base_dir']
    
    # Ensure base_results_dir exists
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Find next available multi_run folder inside base_dir
    multi_run_idx = 1
    while True:
        multi_run_folder = os.path.join(base_results_dir, f'multi_run_{multi_run_idx:03d}')
        if not os.path.exists(multi_run_folder):
            break
        multi_run_idx += 1
    
    os.makedirs(multi_run_folder, exist_ok=True)
    logger.info("Multi-run parent folder: %s", multi_run_folder)
    
    # Save config to multi-run folder
    config_copy_path = os.path.join(multi_run_folder, 'config.yaml')
    import shutil
    shutil.copy(config_path, config_copy_path)
    
    all_run_results = []
    
    for run_number in range(1, num_runs + 1):
        run_seed = base_seed + run_number
        logger.info("\n" + "=" * 80)
        logger.info("Starting run %d/%d (seed=%d)", run_number, num_runs, run_seed)
        logger.info("=" * 80)
        
        # Create individual run folder
        run_folder = os.path.join(multi_run_folder, f'run_{run_number:03d}')
        os.makedirs(run_folder, exist_ok=True)
        
        # Train with current seed
        run_results = train_experiment_6_static(
            npz_path=npz_path,
            config_path=config_path,
            results_folder=run_folder,
            run_idx=run_number,
            seed=run_seed,
        )
        
        all_run_results.append(run_results)
        
        logger.info(
            "Run %d complete: acc=%.4f, macro_f1=%.4f",
            run_number,
            run_results['test_metrics']['accuracy'],
            run_results['test_metrics']['macro_f1'],
        )
        
        # Memory cleanup
        if config['memory'].get('clear_session_after_run', True):
            tf.keras.backend.clear_session()
        if config['memory'].get('force_gc_after_run', True):
            gc.collect()
    
    # Compute aggregated statistics
    aggregated_stats = _compute_aggregation_stats_exp6(all_run_results)
    
    # Save multi-run summary
    _save_multi_run_summary_exp6(all_run_results, aggregated_stats, multi_run_folder)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-RUN EXPERIMENT 6 COMPLETE")
    logger.info("=" * 80)
    logger.info("Test Accuracy: %.4f ± %.4f", 
                aggregated_stats['test_accuracy']['mean'],
                aggregated_stats['test_accuracy']['std'])
    logger.info("Test Macro F1: %.4f ± %.4f",
                aggregated_stats['test_macro_f1']['mean'],
                aggregated_stats['test_macro_f1']['std'])
    logger.info("=" * 80)
    
    return all_run_results, aggregated_stats


__all__ = ['train_experiment_6_static', 'train_experiment_6_multi_run', 
           'train_experiment_6_temporal', 'train_experiment_6_temporal_multi_run']


# ============================================================================
# TEMPORAL POSE TRAINING FUNCTIONS
# ============================================================================


def _filter_temporal_features(
    dataset: List[Tuple], 
    selected_angles: List[str], 
    all_angle_names: List[str]
) -> List[Tuple]:
    """Filter temporal pose features to include only selected angles, then flatten.
    
    This function selects entire timestep sequences for specific angles, then flattens
    the 2D temporal data (T_fixed, num_selected_angles) into 1D vectors for MLP input.
    
    Args:
        dataset (List[Tuple]): Dataset with (label, temporal_features, subject_id, view) tuples.
            temporal_features has shape (T_fixed, num_angles), e.g., (50, 9).
        selected_angles (List[str]): List of angle names to keep (e.g., ['left_elbow', 'right_knee']).
        all_angle_names (List[str]): Complete list of angle names from NPZ file.
    
    Returns:
        List[Tuple]: Filtered dataset with flattened features.
            Features are reshaped from (T_fixed, num_selected_angles) to (T_fixed * num_selected_angles,).
    """
    if not selected_angles or selected_angles == ['all']:
        # Use all angles, just flatten
        filtered_dataset = []
        for item in dataset:
            if len(item) == 4:
                label, features, subject_id, view = item
                # Flatten from (T_fixed, num_angles) to (T_fixed * num_angles,)
                flattened_features = features.flatten()
                filtered_dataset.append((label, flattened_features, subject_id, view))
            else:
                label, features, subject_id = item
                flattened_features = features.flatten()
                filtered_dataset.append((label, flattened_features, subject_id))
        
        return filtered_dataset
    
    # Find indices of selected angles
    angle_indices = []
    for angle_name in selected_angles:
        if angle_name in all_angle_names:
            angle_indices.append(all_angle_names.index(angle_name))
        else:
            logger.warning(f"Angle '{angle_name}' not found in dataset. Available: {all_angle_names}")
    
    if not angle_indices:
        raise ValueError(f"No valid angles selected. Available angles: {all_angle_names}")
    
    # Filter temporal sequences for selected angles
    filtered_dataset = []
    for item in dataset:
        if len(item) == 4:
            label, features, subject_id, view = item
            # Select columns for chosen angles: (T_fixed, num_angles) -> (T_fixed, num_selected)
            filtered_features = features[:, angle_indices]
            # Flatten to 1D vector for MLP: (T_fixed, num_selected) -> (T_fixed * num_selected,)
            flattened_features = filtered_features.flatten()
            filtered_dataset.append((label, flattened_features, subject_id, view))
        else:
            label, features, subject_id = item
            filtered_features = features[:, angle_indices]
            flattened_features = filtered_features.flatten()
            filtered_dataset.append((label, flattened_features, subject_id))
    
    logger.info(
        "Filtered temporal features: %d angles selected (%s) -> shape %s -> %d flattened features",
        len(angle_indices),
        ', '.join([all_angle_names[i] for i in angle_indices]),
        filtered_features.shape,
        flattened_features.shape[0]
    )
    
    return filtered_dataset


def train_experiment_6_temporal(
    npz_path: str,
    config_path: str = 'config/experiment_6_temporal.yaml',
    *,
    results_folder: Optional[str] = None,
    run_idx: Optional[int] = None,
    # Legacy parameters for backward compatibility
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    hidden_sizes: Optional[Tuple[int, ...]] = None,
    dropout: Optional[float] = None,
    lr: Optional[float] = None,
    stratified: Optional[bool] = None,
    max_epochs: Optional[int] = None,
) -> Dict:
    """Train a pose-based MLP baseline with subject-wise stratified splits using temporal features.
    
    Temporal features are flattened from (T_fixed, num_angles) to 1D vectors before feeding to MLP.
    This establishes a baseline for comparing raw temporal sequences vs hand-crafted statistics.

    Args:
        npz_path (str): Path to pose temporal NPZ file (front or side view).
        config_path (str): Path to YAML configuration file. Defaults to 'config/experiment_6_temporal.yaml'.
        results_folder (Optional[str]): Pre-created results folder path (used in multi-run mode).
        run_idx (Optional[int]): Run index for naming (used in multi-run mode).
        
        Legacy parameters (overridden by config if None):
        val_ratio (Optional[float]): Fraction of subjects for validation split.
        test_ratio (Optional[float]): Fraction of subjects for test split.
        seed (Optional[int]): Random seed for reproducibility.
        batch_size (Optional[int]): Batch size for tf.data pipelines.
        hidden_sizes (Optional[Tuple[int, ...]]): Hidden layer widths for the MLP.
        dropout (Optional[float]): Dropout rate after each hidden layer.
        lr (Optional[float]): Learning rate for Adam.
        stratified (Optional[bool]): Ensure all classes appear in train/val/test when possible.
        max_epochs (Optional[int]): Training epochs with early stopping.

    Returns:
        Dict: Training history, label maps, split sizes, and test metrics including macro F1.
    """

    # Load configuration
    config = load_config(config_path)
    
    # Extract hyperparameters from config (or use legacy parameters if provided)
    val_ratio = val_ratio if val_ratio is not None else config['dataset']['val_ratio']
    test_ratio = test_ratio if test_ratio is not None else config['dataset']['test_ratio']
    seed = seed if seed is not None else config['dataset']['random_seed']
    batch_size = batch_size if batch_size is not None else config['training']['batch_size']
    hidden_sizes = hidden_sizes if hidden_sizes is not None else tuple(config['model']['hidden_sizes'])
    dropout = dropout if dropout is not None else config['model']['dropout']
    lr = lr if lr is not None else config['training']['lr']
    stratified = stratified if stratified is not None else config['dataset']['stratified']
    max_epochs = max_epochs if max_epochs is not None else config['training']['max_epochs']
    
    # Setup results folder if not provided (single-run mode)
    if results_folder is None:
        base_results_dir = config['results']['base_dir']
        results_folder, run_idx = setup_results_folder(base_results_dir)
        logger.info("Experiment 6 (temporal) results directory: %s", results_folder)
    
    set_global_seed(seed)

    # Load temporal data
    dataset, pose_summary = load_pose_temporal_data(npz_path)
    
    # Filter and flatten temporal features based on selected angles
    selected_angles = config.get('dataset', {}).get('selected_angles', ['all'])
    if selected_angles and selected_angles != ['all']:
        if 'angle_names' not in pose_summary:
            raise KeyError(
                "NPZ file is missing 'angle_names' field but config specifies selected_angles. "
                "Please regenerate the NPZ file or set selected_angles to ['all']."
            )
        angle_names = pose_summary['angle_names']
        dataset = _filter_temporal_features(dataset, selected_angles, angle_names)
    else:
        # Flatten all temporal features
        dataset = _filter_temporal_features(dataset, ['all'], pose_summary['angle_names'])

    train_samples, val_samples, test_samples, label_to_int = make_pose_split_three_way(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratified=stratified,
    )

    train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test = build_pose_datasets_three_way(
        train_samples,
        val_samples,
        test_samples,
        label_to_int,
        batch_size=batch_size,
    )

    input_dim = X_train.shape[1]
    num_classes = len(label_to_int)
    int_to_label = {v: k for k, v in label_to_int.items()}

    model = _build_mlp(input_dim, num_classes, hidden_sizes, dropout, lr)

    # Build callbacks from config
    callbacks = _build_callbacks(config)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)

    macro_f1 = macro_f1_score(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    per_class_f1 = per_class_f1_scores(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    # Extract hyperparameters summary for saving
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    reduce_lr_cfg = config.get('callbacks', {}).get('reduce_lr_on_plateau', {})
    hyperparameters = {
        'batch_size': batch_size,
        'hidden_sizes': list(hidden_sizes),
        'dropout': dropout,
        'learning_rate': lr,
        'max_epochs': max_epochs,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'stratified': stratified,
        'selected_angles': config.get('dataset', {}).get('selected_angles', ['all']),
        'early_stopping': {
            'enabled': early_cfg.get('enabled', True),
            'monitor': early_cfg.get('monitor', 'val_loss'),
            'patience': early_cfg.get('patience', 10),
            'min_delta': early_cfg.get('min_delta', 0.0),
        },
        'reduce_lr_on_plateau': {
            'enabled': reduce_lr_cfg.get('enabled', False),
            'monitor': reduce_lr_cfg.get('monitor', 'val_loss'),
            'factor': reduce_lr_cfg.get('factor', 0.5),
            'patience': reduce_lr_cfg.get('patience', 15),
            'min_lr': reduce_lr_cfg.get('min_lr', 1e-7),
        },
        'input_dim': input_dim,
        'num_classes': num_classes,
        'feature_type': 'temporal_flattened',
    }

    results = {
        'run_idx': run_idx,
        'seed': seed,
        'hyperparameters': hyperparameters,
        'pose_summary': pose_summary,
        'label_to_int': label_to_int,
        'int_to_label': int_to_label,
        'history': history.history,
        'test_metrics': {
            'loss': float(test_eval['loss']),
            'accuracy': float(test_eval.get('accuracy', 0.0)),
            'macro_f1': macro_f1,
            'per_class_f1': {int(k): float(v) for k, v in per_class_f1.items()},
            'confusion_matrix': conf_matrix.tolist(),
        },
        'train_sizes': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
        },
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
    }

    # Save results to disk
    if config['results'].get('save_metrics', True):
        metrics_path = os.path.join(results_folder, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(
        "Experiment 6 (temporal) complete. Test acc=%.4f, macro F1=%.4f",
        results['test_metrics']['accuracy'],
        results['test_metrics']['macro_f1'],
    )

    return results


def train_experiment_6_temporal_multi_run(
    npz_path: str,
    config_path: str = 'config/experiment_6_temporal.yaml'
) -> Tuple[List[Dict], Dict]:
    """Execute multiple temporal training runs with different seeds and aggregate results.

    Args:
        npz_path (str): Path to pose temporal NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.

    Returns:
        Tuple[List[Dict], Dict]: (all_run_results, aggregated_stats) containing individual
            run metrics and aggregated statistics across all runs.
    """
    config = load_config(config_path)
    
    # Validate multi-run is enabled
    multi_run_cfg = config.get('multi_run', {})
    if not multi_run_cfg.get('enabled', False):
        raise ValueError(
            "Multi-run mode is disabled in config. Set 'multi_run.enabled: true' "
            "or use train_experiment_6_temporal() for single runs."
        )
    
    num_runs = multi_run_cfg.get('num_runs', 30)
    base_seed = multi_run_cfg.get('base_seed', 42)
    save_all_runs = multi_run_cfg.get('save_all_runs', True)
    
    # Create multi-run folder inside base_dir
    base_results_dir = config['results']['base_dir']
    
    # Ensure base_results_dir exists
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Find next available multi_run folder inside base_dir
    multi_run_idx = 1
    while True:
        multi_run_folder = os.path.join(base_results_dir, f'multi_run_{multi_run_idx:03d}')
        if not os.path.exists(multi_run_folder):
            break
        multi_run_idx += 1
    
    os.makedirs(multi_run_folder, exist_ok=True)
    logger.info("Multi-run parent folder: %s", multi_run_folder)
    
    # Save config to multi-run folder
    config_copy_path = os.path.join(multi_run_folder, 'config.yaml')
    import shutil
    shutil.copy(config_path, config_copy_path)
    
    all_run_results = []
    
    for run_number in range(1, num_runs + 1):
        run_seed = base_seed + run_number
        logger.info("\n" + "=" * 80)
        logger.info("Starting run %d/%d (seed=%d)", run_number, num_runs, run_seed)
        logger.info("=" * 80)
        
        # Create individual run folder
        run_folder = os.path.join(multi_run_folder, f'run_{run_number:03d}')
        os.makedirs(run_folder, exist_ok=True)
        
        # Train with current seed
        run_results = train_experiment_6_temporal(
            npz_path=npz_path,
            config_path=config_path,
            results_folder=run_folder,
            run_idx=run_number,
            seed=run_seed,
        )
        
        all_run_results.append(run_results)
        
        logger.info(
            "Run %d complete: acc=%.4f, macro_f1=%.4f",
            run_number,
            run_results['test_metrics']['accuracy'],
            run_results['test_metrics']['macro_f1'],
        )
        
        # Memory cleanup
        if config['memory'].get('clear_session_after_run', True):
            tf.keras.backend.clear_session()
        if config['memory'].get('force_gc_after_run', True):
            gc.collect()
    
    # Compute aggregated statistics
    logger.info("\n" + "=" * 80)
    logger.info("Computing aggregated statistics across %d runs", num_runs)
    logger.info("=" * 80)
    
    aggregated_stats = _compute_aggregation_stats_exp6(all_run_results)
    
    # Save multi-run summary
    _save_multi_run_summary_exp6(all_run_results, aggregated_stats, multi_run_folder)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-RUN EXPERIMENT 6 (TEMPORAL) COMPLETE")
    logger.info("=" * 80)
    logger.info("Test Accuracy: %.4f ± %.4f", 
                aggregated_stats['test_accuracy']['mean'],
                aggregated_stats['test_accuracy']['std'])
    logger.info("Test Macro F1: %.4f ± %.4f",
                aggregated_stats['test_macro_f1']['mean'],
                aggregated_stats['test_macro_f1']['std'])
    logger.info("Results saved to: %s", multi_run_folder)
    logger.info("=" * 80)
    
    return all_run_results, aggregated_stats

