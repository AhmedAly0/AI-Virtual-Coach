"""Experiment 7: LSTM-based pose classification for exercise recognition."""

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
from src.data.dataset_builder import make_pose_split_three_way
from src.utils.io_utils import set_global_seed, load_config, setup_results_folder
from src.utils.metrics import macro_f1_score, per_class_f1_scores

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
logger.propagate = True


# ============================================================================
# MODEL BUILDING
# ============================================================================


def _build_lstm(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.2,
    bidirectional: bool = False,
    lr: float = 0.001
) -> tf.keras.Model:
    """Build LSTM model for sequential pose data.

    Args:
        input_shape (Tuple[int, int]): Shape of input sequences (timesteps, features).
        num_classes (int): Number of exercise classes.
        lstm_units (int): Number of units in LSTM layer.
        dense_units (int): Number of units in dense layer after LSTM.
        dropout (float): Dropout rate for dense layers.
        recurrent_dropout (float): Dropout rate within LSTM recurrent connections.
        bidirectional (bool): Whether to use bidirectional LSTM.
        lr (float): Learning rate for Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras LSTM model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    
    # LSTM layer
    lstm_layer = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    )
    
    if bidirectional:
        lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)
    
    model.add(lstm_layer)
    
    # Dense layers
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    
    return model


# ============================================================================
# FEATURE PROCESSING FOR TEMPORAL DATA
# ============================================================================


def _filter_temporal_features_for_lstm(
    dataset: List[Tuple],
    selected_angles: List[str],
    all_angle_names: List[str]
) -> List[Tuple]:
    """Filter temporal pose features by angle WITHOUT flattening (keep 2D shape for LSTM).
    
    Args:
        dataset (List[Tuple]): Dataset with (label, temporal_features, subject_id, view) tuples.
            temporal_features has shape (T_fixed, num_angles), e.g., (50, 9).
        selected_angles (List[str]): List of angle names to keep.
        all_angle_names (List[str]): Complete list of angle names from NPZ file.
    
    Returns:
        List[Tuple]: Filtered dataset with shape (T_fixed, num_selected_angles).
    """
    if not selected_angles or selected_angles == ['all']:
        # Use all angles, keep 2D shape
        return dataset
    
    # Find indices of selected angles
    angle_indices = []
    for angle_name in selected_angles:
        if angle_name in all_angle_names:
            angle_indices.append(all_angle_names.index(angle_name))
        else:
            logger.warning(f"Angle '{angle_name}' not found in dataset. Available: {all_angle_names}")
    
    if not angle_indices:
        raise ValueError(f"No valid angles selected. Available angles: {all_angle_names}")
    
    # Filter temporal sequences for selected angles (keep 2D)
    filtered_dataset = []
    for item in dataset:
        if len(item) == 4:
            label, features, subject_id, view = item
            # Select columns for chosen angles: (T_fixed, num_angles) -> (T_fixed, num_selected)
            filtered_features = features[:, angle_indices]
            filtered_dataset.append((label, filtered_features, subject_id, view))
        else:
            label, features, subject_id = item
            filtered_features = features[:, angle_indices]
            filtered_dataset.append((label, filtered_features, subject_id))
    
    logger.info(
        "Filtered temporal features for LSTM: %d angles selected (%s) -> shape %s",
        len(angle_indices),
        ', '.join([all_angle_names[i] for i in angle_indices]),
        filtered_features.shape
    )
    
    return filtered_dataset


# ============================================================================
# FEATURE PROCESSING FOR STATIC DATA (RESHAPE FOR LSTM)
# ============================================================================


def _filter_and_reshape_static_for_lstm(
    dataset: List[Tuple],
    selected_angles: List[str],
    all_angle_names: List[str]
) -> List[Tuple]:
    """Filter static pose features and reshape for LSTM input.
    
    Reshapes static features from (num_angles * 5_stats,) to (num_angles, 5_stats),
    treating angles as timesteps and statistics as features per timestep.
    
    Args:
        dataset (List[Tuple]): Dataset with (label, static_features, subject_id, view) tuples.
            static_features has shape (num_angles * 5,), e.g., (45,) for 9 angles.
        selected_angles (List[str]): List of angle names to keep.
        all_angle_names (List[str]): Complete list of angle names from NPZ file.
    
    Returns:
        List[Tuple]: Reshaped dataset with shape (num_selected_angles, 5).
    """
    num_stats = 5  # mean, std, min, max, range
    
    if not selected_angles or selected_angles == ['all']:
        selected_angles = all_angle_names
    
    # Find indices of selected angles
    angle_indices = []
    for angle_name in selected_angles:
        if angle_name in all_angle_names:
            angle_indices.append(all_angle_names.index(angle_name))
        else:
            logger.warning(f"Angle '{angle_name}' not found in dataset. Available: {all_angle_names}")
    
    if not angle_indices:
        raise ValueError(f"No valid angles selected. Available angles: {all_angle_names}")
    
    # Compute feature indices for selected angles (each angle has 5 features)
    feature_indices = []
    for angle_idx in angle_indices:
        start_col = angle_idx * num_stats
        feature_indices.extend(range(start_col, start_col + num_stats))
    
    num_selected_angles = len(angle_indices)
    
    # Filter and reshape for each sample
    reshaped_dataset = []
    for item in dataset:
        if len(item) == 4:
            label, features, subject_id, view = item
            # Filter features for selected angles
            filtered_features = features[feature_indices]
            # Reshape from (num_selected_angles * 5,) to (num_selected_angles, 5)
            reshaped_features = filtered_features.reshape(num_selected_angles, num_stats)
            reshaped_dataset.append((label, reshaped_features, subject_id, view))
        else:
            label, features, subject_id = item
            filtered_features = features[feature_indices]
            reshaped_features = filtered_features.reshape(num_selected_angles, num_stats)
            reshaped_dataset.append((label, reshaped_features, subject_id))
    
    logger.info(
        "Reshaped static features for LSTM: %d angles selected (%s) -> shape %s (angles as timesteps)",
        num_selected_angles,
        ', '.join([all_angle_names[i] for i in angle_indices]),
        reshaped_features.shape
    )
    
    return reshaped_dataset


# ============================================================================
# DATASET BUILDING FOR LSTM (2D SEQUENTIAL INPUT)
# ============================================================================


def _build_lstm_datasets(
    train_samples: List[Tuple],
    val_samples: List[Tuple],
    test_samples: List[Tuple],
    label_to_int: Dict[str, int],
    batch_size: int = 64,
    standardize: bool = True
) -> Tuple:
    """Build tf.data datasets for LSTM with 2D sequential inputs.
    
    Args:
        train_samples: List of (label, features_2d, subject_id, view) tuples.
        val_samples: Validation samples.
        test_samples: Test samples.
        label_to_int: Mapping from label names to integers.
        batch_size: Batch size for tf.data pipelines.
        standardize: Whether to standardize features using training set statistics.
    
    Returns:
        Tuple: (train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Convert to numpy arrays
    X_train = np.array([s[1] for s in train_samples], dtype=np.float32)
    y_train = np.array([label_to_int[s[0]] for s in train_samples], dtype=np.int32)
    
    X_val = np.array([s[1] for s in val_samples], dtype=np.float32)
    y_val = np.array([label_to_int[s[0]] for s in val_samples], dtype=np.int32)
    
    X_test = np.array([s[1] for s in test_samples], dtype=np.float32)
    y_test = np.array([label_to_int[s[0]] for s in test_samples], dtype=np.int32)
    
    # Standardize using training set statistics (per feature across all timesteps)
    if standardize:
        # Compute mean and std across samples and timesteps, per feature
        # X_train shape: (N, timesteps, features)
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
    
    # Build tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# TEMPORAL TRAINING FUNCTIONS
# ============================================================================


def train_experiment_7_temporal(
    npz_path: str,
    config_path: str = 'config/experiment_7_temporal.yaml',
    *,
    results_folder: Optional[str] = None,
    run_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Train LSTM on temporal pose features (native sequential input).

    Args:
        npz_path (str): Path to pose temporal NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.
        results_folder (Optional[str]): Pre-created results folder path (used in multi-run mode).
        run_idx (Optional[int]): Run index for naming (used in multi-run mode).
        seed (Optional[int]): Random seed override.

    Returns:
        Dict: Training history, label maps, split sizes, and test metrics including macro F1.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract hyperparameters from config
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    seed = seed if seed is not None else config['dataset']['random_seed']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    max_epochs = config['training']['max_epochs']
    stratified = config['dataset']['stratified']
    
    # LSTM-specific parameters
    lstm_units = config['model']['lstm_units']
    dense_units = config['model'].get('dense_units', 64)
    dropout = config['model']['dropout']
    recurrent_dropout = config['model']['recurrent_dropout']
    bidirectional = config['model'].get('bidirectional', False)
    
    # Setup results folder if not provided (single-run mode)
    if results_folder is None:
        base_results_dir = config['results']['base_dir']
        results_folder, run_idx = setup_results_folder(base_results_dir)
        logger.info("Experiment 7 (LSTM temporal) results directory: %s", results_folder)
    
    set_global_seed(seed)
    
    # Load temporal data (shape: N, T_fixed, num_angles)
    dataset, pose_summary = load_pose_temporal_data(npz_path)
    
    # Filter temporal features by selected angles (keep 2D for LSTM)
    selected_angles = config.get('dataset', {}).get('selected_angles', ['all'])
    if selected_angles and selected_angles != ['all']:
        if 'angle_names' not in pose_summary:
            raise KeyError(
                "NPZ file is missing 'angle_names' field but config specifies selected_angles. "
                "Please regenerate the NPZ file or set selected_angles to ['all']."
            )
        angle_names = pose_summary['angle_names']
        dataset = _filter_temporal_features_for_lstm(dataset, selected_angles, angle_names)
    
    # Split data
    train_samples, val_samples, test_samples, label_to_int = make_pose_split_three_way(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratified=stratified,
    )
    
    # Build tf.data datasets for LSTM
    train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test = _build_lstm_datasets(
        train_samples,
        val_samples,
        test_samples,
        label_to_int,
        batch_size=batch_size,
    )
    
    # Get input shape (timesteps, features)
    input_shape = X_train.shape[1:]  # e.g., (50, 8)
    num_classes = len(label_to_int)
    int_to_label = {v: k for k, v in label_to_int.items()}
    
    # Build LSTM model
    model = _build_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        bidirectional=bidirectional,
        lr=lr
    )
    
    # Build callbacks from config
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    callbacks: List[tf.keras.callbacks.Callback] = []
    if early_cfg.get('enabled', True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get('monitor', 'val_loss'),
                patience=early_cfg.get('patience', 15),
                restore_best_weights=early_cfg.get('restore_best_weights', True),
                min_delta=early_cfg.get('min_delta', 0.0),
                verbose=1,
            )
        )
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on test set
    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)
    
    macro_f1 = macro_f1_score(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    per_class_f1 = per_class_f1_scores(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    
    # Extract hyperparameters summary for saving
    hyperparameters = {
        'batch_size': batch_size,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'bidirectional': bidirectional,
        'learning_rate': lr,
        'max_epochs': max_epochs,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'stratified': stratified,
        'selected_angles': config.get('dataset', {}).get('selected_angles', ['all']),
        'early_stopping': {
            'enabled': early_cfg.get('enabled', True),
            'monitor': early_cfg.get('monitor', 'val_loss'),
            'patience': early_cfg.get('patience', 15),
            'min_delta': early_cfg.get('min_delta', 0.0),
        },
        'input_shape': list(input_shape),
        'num_classes': num_classes,
        'feature_type': 'temporal_sequential',
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

    return results


# ============================================================================
# STATIC TRAINING FUNCTIONS
# ============================================================================


def train_experiment_7_static(
    npz_path: str,
    config_path: str = 'config/experiment_7_static.yaml',
    *,
    results_folder: Optional[str] = None,
    run_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Train LSTM on static pose features (reshaped to sequential input).
    
    Static features are reshaped from (num_angles * 5,) to (num_angles, 5),
    treating angles as timesteps and 5 statistics as features per timestep.

    Args:
        npz_path (str): Path to pose static NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.
        results_folder (Optional[str]): Pre-created results folder path.
        run_idx (Optional[int]): Run index for naming.
        seed (Optional[int]): Random seed override.

    Returns:
        Dict: Training history, label maps, split sizes, and test metrics.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract hyperparameters
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    seed = seed if seed is not None else config['dataset']['random_seed']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    max_epochs = config['training']['max_epochs']
    stratified = config['dataset']['stratified']
    
    # LSTM-specific parameters
    lstm_units = config['model']['lstm_units']
    dense_units = config['model'].get('dense_units', 64)
    dropout = config['model']['dropout']
    recurrent_dropout = config['model']['recurrent_dropout']
    bidirectional = config['model'].get('bidirectional', False)
    
    # Setup results folder if not provided
    if results_folder is None:
        base_results_dir = config['results']['base_dir']
        results_folder, run_idx = setup_results_folder(base_results_dir)
        logger.info("Experiment 7 (LSTM static) results directory: %s", results_folder)
    
    set_global_seed(seed)
    
    # Load static data
    dataset, pose_summary = load_pose_data(npz_path)
    
    # Filter and reshape static features for LSTM
    selected_angles = config.get('dataset', {}).get('selected_angles', ['all'])
    if 'angle_names' not in pose_summary:
        raise KeyError(
            "NPZ file is missing 'angle_names' field. "
            "Please regenerate the NPZ file."
        )
    angle_names = pose_summary['angle_names']
    
    # Reshape static to (num_angles, 5) for LSTM
    dataset = _filter_and_reshape_static_for_lstm(dataset, selected_angles, angle_names)
    logger.info("Using selected angles: %s", selected_angles)
    
    # Split data
    train_samples, val_samples, test_samples, label_to_int = make_pose_split_three_way(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratified=stratified,
    )
    
    # Build tf.data datasets for LSTM
    train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test = _build_lstm_datasets(
        train_samples,
        val_samples,
        test_samples,
        label_to_int,
        batch_size=batch_size,
    )
    
    # Get input shape (num_angles, 5_stats)
    input_shape = X_train.shape[1:]  # e.g., (8, 5)
    num_classes = len(label_to_int)
    int_to_label = {v: k for k, v in label_to_int.items()}
    
    # Build LSTM model
    model = _build_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        bidirectional=bidirectional,
        lr=lr
    )
    
    # Build callbacks from config
    early_cfg = config.get('callbacks', {}).get('early_stopping', {})
    callbacks: List[tf.keras.callbacks.Callback] = []
    if early_cfg.get('enabled', True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get('monitor', 'val_loss'),
                patience=early_cfg.get('patience', 15),
                restore_best_weights=early_cfg.get('restore_best_weights', True),
                min_delta=early_cfg.get('min_delta', 0.0),
                verbose=1,
            )
        )
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on test set
    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)
    
    macro_f1 = macro_f1_score(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    per_class_f1 = per_class_f1_scores(y_true=y_test, y_pred=y_pred, num_classes=num_classes)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    
    # Extract hyperparameters summary
    hyperparameters = {
        'batch_size': batch_size,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'bidirectional': bidirectional,
        'learning_rate': lr,
        'max_epochs': max_epochs,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'stratified': stratified,
        'selected_angles': config.get('dataset', {}).get('selected_angles', ['all']),
        'early_stopping': {
            'enabled': early_cfg.get('enabled', True),
            'monitor': early_cfg.get('monitor', 'val_loss'),
            'patience': early_cfg.get('patience', 15),
            'min_delta': early_cfg.get('min_delta', 0.0),
        },
        'input_shape': list(input_shape),
        'num_classes': num_classes,
        'feature_type': 'static_reshaped',
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

    return results


# ============================================================================
# AGGREGATION UTILITIES
# ============================================================================


def _compute_aggregation_stats_exp7(all_run_results: List[Dict]) -> Dict:
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


def _save_multi_run_summary_exp7(
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
        f.write("EXPERIMENT 7 (LSTM) MULTI-RUN SUMMARY\n")
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


# ============================================================================
# MULTI-RUN WRAPPERS
# ============================================================================


def train_experiment_7_temporal_multi_run(
    npz_path: str,
    config_path: str = 'config/experiment_7_temporal.yaml'
) -> Tuple[List[Dict], Dict]:
    """Execute multiple LSTM temporal training runs with different seeds.

    Args:
        npz_path (str): Path to pose temporal NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.

    Returns:
        Tuple[List[Dict], Dict]: (all_run_results, aggregated_stats)
    """
    config = load_config(config_path)
    
    # Validate multi-run is enabled
    multi_run_cfg = config.get('multi_run', {})
    if not multi_run_cfg.get('enabled', False):
        raise ValueError(
            "Multi-run mode is disabled in config. Set 'multi_run.enabled: true' "
            "or use train_experiment_7_temporal() for single runs."
        )
    
    num_runs = multi_run_cfg.get('num_runs', 30)
    base_seed = multi_run_cfg.get('base_seed', 42)
    
    # Create multi-run folder inside base_dir
    base_results_dir = config['results']['base_dir']
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Find next available multi_run folder
    multi_run_idx = 1
    while True:
        multi_run_folder = os.path.join(base_results_dir, f'multi_run_{multi_run_idx:03d}')
        if not os.path.exists(multi_run_folder):
            break
        multi_run_idx += 1
    
    os.makedirs(multi_run_folder, exist_ok=True)
    logger.info("Multi-run parent folder: %s", multi_run_folder)
    
    # Save config to multi-run folder
    import shutil
    config_copy_path = os.path.join(multi_run_folder, 'config.yaml')
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
        run_results = train_experiment_7_temporal(
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
    aggregated_stats = _compute_aggregation_stats_exp7(all_run_results)
    _save_multi_run_summary_exp7(all_run_results, aggregated_stats, multi_run_folder)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-RUN EXPERIMENT 7 (LSTM TEMPORAL) COMPLETE")
    logger.info("=" * 80)
    logger.info("Test Accuracy: %.4f ± %.4f", 
                aggregated_stats['test_accuracy']['mean'],
                aggregated_stats['test_accuracy']['std'])
    logger.info("Test Macro F1: %.4f ± %.4f",
                aggregated_stats['test_macro_f1']['mean'],
                aggregated_stats['test_macro_f1']['std'])
    logger.info("=" * 80)
    
    return all_run_results, aggregated_stats


def train_experiment_7_static_multi_run(
    npz_path: str,
    config_path: str = 'config/experiment_7_static.yaml'
) -> Tuple[List[Dict], Dict]:
    """Execute multiple LSTM static training runs with different seeds.

    Args:
        npz_path (str): Path to pose static NPZ file (front or side view).
        config_path (str): Path to YAML configuration file.

    Returns:
        Tuple[List[Dict], Dict]: (all_run_results, aggregated_stats)
    """
    config = load_config(config_path)
    
    # Validate multi-run is enabled
    multi_run_cfg = config.get('multi_run', {})
    if not multi_run_cfg.get('enabled', False):
        raise ValueError(
            "Multi-run mode is disabled in config. Set 'multi_run.enabled: true' "
            "or use train_experiment_7_static() for single runs."
        )
    
    num_runs = multi_run_cfg.get('num_runs', 30)
    base_seed = multi_run_cfg.get('base_seed', 42)
    
    # Create multi-run folder inside base_dir
    base_results_dir = config['results']['base_dir']
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Find next available multi_run folder
    multi_run_idx = 1
    while True:
        multi_run_folder = os.path.join(base_results_dir, f'multi_run_{multi_run_idx:03d}')
        if not os.path.exists(multi_run_folder):
            break
        multi_run_idx += 1
    
    os.makedirs(multi_run_folder, exist_ok=True)
    logger.info("Multi-run parent folder: %s", multi_run_folder)
    
    # Save config to multi-run folder
    import shutil
    config_copy_path = os.path.join(multi_run_folder, 'config.yaml')
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
        run_results = train_experiment_7_static(
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
    aggregated_stats = _compute_aggregation_stats_exp7(all_run_results)
    _save_multi_run_summary_exp7(all_run_results, aggregated_stats, multi_run_folder)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-RUN EXPERIMENT 7 (LSTM STATIC) COMPLETE")
    logger.info("=" * 80)
    logger.info("Test Accuracy: %.4f ± %.4f", 
                aggregated_stats['test_accuracy']['mean'],
                aggregated_stats['test_accuracy']['std'])
    logger.info("Test Macro F1: %.4f ± %.4f",
                aggregated_stats['test_macro_f1']['mean'],
                aggregated_stats['test_macro_f1']['std'])
    logger.info("=" * 80)
    
    return all_run_results, aggregated_stats


__all__ = [
    'train_experiment_7_temporal',
    'train_experiment_7_temporal_multi_run',
    'train_experiment_7_static',
    'train_experiment_7_static_multi_run',
]
