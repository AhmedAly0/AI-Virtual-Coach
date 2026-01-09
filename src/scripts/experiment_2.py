"""
Experiment 2: 3-Stage Progressive Unfreezing with Blind Training

Training Strategy:
- Stage 1: Train with frozen backbone (no validation monitoring)
- Stage 2: Unfreeze bottom 10% of backbone layers
- Stage 3: Unfreeze bottom 30% of backbone layers

Differences from Experiment 1:
- NO validation monitoring during training (blind training)
- 3-stage progressive unfreezing strategy
- Enhanced augmentation (flip + translation + rotation + zoom + brightness)
- Architecture-specific classification heads
- Per-backbone epoch configuration
"""

import os
import gc
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Tuple, Dict, Optional
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from ..data import make_split, build_datasets
from ..models import build_model_for_backbone_v2
from ..utils import (
    setup_results_folder_for_backbone,
    save_experiment_summary,
    save_backbone_config,
    save_training_curves_train_only
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
logger.propagate = True


def load_config(config_path: str) -> Dict:
    """
    Loads experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: The loaded configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_run_progressive(
    run_idx: int,
    dataset: List[Tuple],
    config: Dict,
    results_folder: str,
    backbone: str
) -> Dict:
    """
    Train with 3-stage progressive unfreezing - NO validation monitoring (pure blind training).
    
    Stage 1: Train with frozen backbone
    Stage 2: Unfreeze bottom 10% of backbone layers
    Stage 3: Unfreeze bottom 30% of backbone layers
    
    Args:
        run_idx (int): Run index for tracking
        dataset (List[Tuple]): Full dataset
        config (Dict): Loaded configuration from YAML file
        results_folder (str): Results directory
        backbone (str): Backbone architecture
        
    Returns:
        Dict: Training results with metrics and histories
    """
    logger.info(f"=" * 80)
    logger.info(f"Run {run_idx} - {backbone}")
    logger.info(f"=" * 80)
    
    # Extract parameters from config
    test_ratio = config['dataset']['test_ratio']
    img_size = config['model']['img_size']
    num_classes = config['model']['num_classes']
    batch_size = config['training']['batch_size']
    initial_lr = config['training']['initial_lr']
    fine_tune_lr = config['training']['fine_tune_lr']
    augment_config = config['augmentation']
    unfreeze_stage_1_percent = config['training']['unfreeze_stage_1_percent']
    unfreeze_stage_2_percent = config['training']['unfreeze_stage_2_percent']
    clear_session = config['memory']['clear_session_after_run']
    force_gc = config['memory']['force_gc_after_run']
    
    # Get backbone-specific epoch configuration
    epochs_cfg = config['epochs'].get(backbone, {'frozen': 20, 'stage1': 10, 'stage2': 20})
    logger.info(f"Epoch config: {epochs_cfg}")
    
    # Save config
    save_backbone_config(backbone, config, results_folder)
    
    # Split & build
    train_samples, test_samples, label_to_int = make_split(dataset, test_ratio, seed=42)
    model, base_model, preprocess_fn = build_model_for_backbone_v2(backbone, img_size, num_classes)
    train_ds, test_ds, X_train, y_train, X_test, y_test = build_datasets(
        train_samples, test_samples, label_to_int, batch_size, img_size, preprocess_fn, augment_config
    )
    
    # ========== STAGE 1: Frozen - NO validation monitoring ==========
    logger.info(f"Stage 1/3: Frozen ({epochs_cfg['frozen']} epochs)")
    
    callbacks_frozen = [
        ReduceLROnPlateau(
            monitor='loss',  # Monitor TRAIN loss only
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    hist_frozen = model.fit(
        train_ds,
        # NO validation_data - blind training
        epochs=epochs_cfg['frozen'],
        callbacks=callbacks_frozen,
        verbose=0
    )
    logger.info(f"  Train acc: {hist_frozen.history['accuracy'][-1]:.4f}")
    
    # ========== STAGE 2: 10% unfrozen ==========
    logger.info(f"Stage 2/3: 10% unfrozen ({epochs_cfg['stage1']} epochs)")
    total_layers = len(base_model.layers)
    N_10 = int(unfreeze_stage_1_percent * total_layers)
    
    for layer in base_model.layers[-N_10:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_stage1 = [
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    hist_stage1 = model.fit(
        train_ds,
        # NO validation_data
        epochs=epochs_cfg['stage1'],
        callbacks=callbacks_stage1,
        verbose=0
    )
    logger.info(f"  Train acc: {hist_stage1.history['accuracy'][-1]:.4f}")
    
    # ========== STAGE 3: 30% unfrozen ==========
    logger.info(f"Stage 3/3: 30% unfrozen ({epochs_cfg['stage2']} epochs)")
    N_30 = int(unfreeze_stage_2_percent * total_layers)
    
    for layer in base_model.layers[-N_30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_fine = [
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    hist_fine = model.fit(
        train_ds,
        # NO validation_data
        epochs=epochs_cfg['stage2'],
        callbacks=callbacks_fine,
        verbose=0
    )
    logger.info(f"  Train acc: {hist_fine.history['accuracy'][-1]:.4f}")
    
    # ========== Evaluation ==========
    # Evaluate ONCE on test set (only after all training is done)
    logger.info("Evaluating model...")
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    logger.info(f"Final: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # Confusion matrix
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred_classes, num_classes=num_classes).numpy()
    
    # Save confusion matrix plot
    plots_dir = os.path.join(results_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CM - Run {run_idx} ({backbone})')
    plt.ylabel('True')
    plt.xlabel('Pred')
    cm_path = os.path.join(plots_dir, f'run_{run_idx:03d}_cm.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {cm_path}")
    
    # Save training curves (train only, no validation)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    all_loss = (
        hist_frozen.history['loss'] +
        hist_stage1.history['loss'] +
        hist_fine.history['loss']
    )
    axes[0].plot(all_loss, color='blue')
    axes[0].set_title('Training Loss (3 Stages)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].axvline(epochs_cfg['frozen'], color='red', linestyle='--', label='Stage 1→2')
    axes[0].axvline(epochs_cfg['frozen'] + epochs_cfg['stage1'], color='orange', linestyle='--', label='Stage 2→3')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    all_acc = (
        hist_frozen.history['accuracy'] +
        hist_stage1.history['accuracy'] +
        hist_fine.history['accuracy']
    )
    axes[1].plot(all_acc, color='green')
    axes[1].set_title('Training Accuracy (3 Stages)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].axvline(epochs_cfg['frozen'], color='red', linestyle='--', label='Stage 1→2')
    axes[1].axvline(epochs_cfg['frozen'] + epochs_cfg['stage1'], color='orange', linestyle='--', label='Stage 2→3')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    curves_path = os.path.join(plots_dir, f'run_{run_idx:03d}_training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved: {curves_path}")
    
    # Save model weights
    model_dir = os.path.join(results_folder, 'models', f'run_{run_idx:03d}')
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, "model.weights.h5")
    model.save_weights(final_model_path)
    logger.info(f"Model weights saved: {final_model_path}")
    
    # Save metrics
    metrics_path = os.path.join(results_folder, "metrics", f"run_{run_idx:03d}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Run {run_idx} Results:\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Epochs: frozen={epochs_cfg['frozen']}, stage1={epochs_cfg['stage1']}, stage2={epochs_cfg['stage2']}\n")
        f.write(f"Train Accuracy: {float(train_acc):.4f}\n")
        f.write(f"Test Accuracy: {float(test_acc):.4f}\n")
        f.write(f"Train Loss: {float(train_loss):.4f}\n")
        f.write(f"Test Loss: {float(test_loss):.4f}\n")
        f.write(f"Final Weights: {final_model_path}\n")
    
    logger.info(f"Metrics saved: {metrics_path}")
    
    # ========== Memory Management ==========
    if CONFIG.get('CLEAR_SESSION_AFTER_RUN'):
        logger.info("Clearing Keras session...")
        K.clear_session()
    
    if CONFIG.get('FORCE_GC_AFTER_RUN'):
        logger.info("Running garbage collection...")
        gc.collect()
    
    logger.info(f"Run {run_idx} completed successfully")
    logger.info("=" * 80)
    
    return {
        'run_idx': run_idx,
        'backbone': backbone,
        'history_frozen': hist_frozen.history,
        'history_stage1': hist_stage1.history,
        'history_fine': hist_fine.history,
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'confusion_matrix': cm,
        'num_train_samples': len(train_samples),
        'num_test_samples': len(test_samples),
        'epochs_config': epochs_cfg,
        'final_weights': final_model_path,
    }


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def train_experiment_2(
    dataset: List[Tuple],
    backbones: List[str],
    config_path: str,
    num_runs: int = 5
) -> Dict[str, List[Dict]]:
    """
    Run Experiment 2 (3-stage progressive unfreezing with blind training).
    
    Args:
        dataset (List[Tuple]): Full dataset
        backbones (List[str]): List of backbone architectures
        config_path (str): Path to the YAML configuration file
        num_runs (int): Number of runs per backbone
        
    Returns:
        Dict[str, List[Dict]]: Results for each backbone
    """
    # Load configuration from YAML file
    config = load_config(config_path)
    
    # Extract parameters from config
    base_results_dir = config['results']['base_dir']
    
    all_backbone_results = {}
    
    for backbone in backbones:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRAINING BACKBONE: {backbone.upper()}")
        logger.info(f"{'=' * 80}\n")
        
        # Setup results folder for this backbone
        results_folder, _ = setup_results_folder_for_backbone(backbone, base_results_dir)
        
        # Run multiple training runs
        backbone_results = []
        for run_idx in range(num_runs):
            result = train_one_run_progressive(
                run_idx=run_idx,
                dataset=dataset,
                config=config,
                results_folder=results_folder,
                backbone=backbone
            )
            backbone_results.append(result)
        
        # Save summary
        save_experiment_summary(backbone_results, results_folder, 0, config['dataset']['test_ratio'])
        
        all_backbone_results[backbone] = backbone_results
        
        logger.info(f"\nCompleted {backbone}: {len(backbone_results)} runs\n")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("EXPERIMENT 2 COMPLETED")
    logger.info(f"{'=' * 80}\n")
    
    return all_backbone_results
