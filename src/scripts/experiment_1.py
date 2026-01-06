"""
Experiment 1: 2-Phase Transfer Learning with Validation Monitoring

This script runs a 2-phase transfer learning experiment. It is designed to be
configured via a YAML file, which separates hyperparameters from the core
training logic.

Training Strategy (as defined in config):
- Phase 1: Train with a frozen backbone.
- Phase 2: Unfreeze the entire model and fine-tune at a lower learning rate.
- Callbacks like EarlyStopping and ReduceLROnPlateau are used with a
  validation set for monitoring.
"""

import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
import logging
from typing import List, Tuple, Dict, Optional
from tensorflow.keras import backend as K

from ..data import make_split, build_datasets
from ..models import build_model_for_backbone, get_callbacks
from ..utils import (
    setup_results_folder,
    setup_results_folder_for_backbone,
    save_experiment_summary,
    save_backbone_config
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
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

def train_one_run(
    run_idx: int,
    dataset: List[Tuple],
    config: Dict,
    results_folder: str = 'results',
    backbone: str = 'resnet50',
) -> Dict:
    """
    Train a single run with 2-phase training (frozen + fine-tuned).
    Uses validation monitoring with EarlyStopping and ReduceLROnPlateau.
    
    Phase 1: Train with frozen backbone (10 epochs)
    Phase 2: Unfreeze and fine-tune entire model (50 epochs)
    
    Args:
        run_idx (int): Run index for tracking
        dataset (List[Tuple]): The full dataset of image paths and labels.
        config (Dict): The experiment configuration dictionary.
        results_folder (str): Results directory
        backbone (str): Backbone architecture
        
    Returns:
        Dict: Training results with metrics and histories
    """
    logger.info(f"=" * 80)
    logger.info(f"Starting Run {run_idx} with backbone: {backbone}")
    logger.info(f"=" * 80)
    
    # Extract parameters from config
    img_size = config['model']['img_size']
    num_classes = config['model']['num_classes']
    batch_size = config['training']['batch_size']
    test_ratio = config['dataset']['test_ratio']
    augment_config = config['augmentation']
    frozen_epochs = config['training']['frozen_epochs']
    fine_tune_epochs = config['training']['fine_tune_epochs']
    fine_tune_lr = config['training']['fine_tune_lr']
    clear_session = config['memory']['clear_session_after_run']
    force_gc = config['memory']['force_gc_after_run']

    
    # Split data
    train_samples, test_samples, label_to_int = make_split(dataset, test_ratio, seed=42)
    
    # Build model
    model, preprocess_fn = build_model_for_backbone(backbone, img_size, num_classes, head_type=config['model']['classification_head'])
    
    # Build datasets
    train_ds, test_ds, X_train, y_train, X_test, y_test = build_datasets(
        train_samples, test_samples, label_to_int, batch_size, img_size, 
        preprocess_fn, augment_config
    )
    
    # ========== PHASE 1: Frozen Training ==========
    logger.info("Phase 1: Training with frozen backbone...")
    
    checkpoint_path_frozen = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_frozen.keras')
    callbacks_frozen = get_callbacks(checkpoint_path_frozen)
    
    hist_frozen = model.fit(
        train_ds,
        validation_data=test_ds,  # Validation monitored
        epochs=frozen_epochs,
        callbacks=callbacks_frozen,
        verbose=1
    )
    
    logger.info("Phase 1 completed")
    
    # ========== PHASE 2: Fine-tuning ==========
    logger.info("Phase 2: Fine-tuning entire model...")
    
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_path_fine = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_fine.keras')
    callbacks_fine = get_callbacks(checkpoint_path_fine)
    
    hist_fine = model.fit(
        train_ds,
        validation_data=test_ds,  # Validation monitored
        epochs=fine_tune_epochs,
        callbacks=callbacks_fine,
        verbose=1
    )
    
    logger.info("Phase 2 completed")
    
    # ========== Evaluation ==========
    logger.info("Evaluating model...")
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred_classes, num_classes=num_classes).numpy()
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Run {run_idx}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(results_folder, 'plots', f'run_{run_idx:03d}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {cm_path}")
    
    # Save model weights
    final_model_dir = os.path.join(results_folder, 'models', f'run_{run_idx:03d}')
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "model.weights.h5")
    model.save_weights(final_model_path)
    logger.info(f"Model weights saved: {final_model_path}")
    
    # Save metrics
    metrics_path = os.path.join(results_folder, "metrics", f"run_{run_idx:03d}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Run {run_idx} Results:\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Train Accuracy: {float(train_acc):.4f}\n")
        f.write(f"Test Accuracy: {float(test_acc):.4f}\n")
        f.write(f"Train Loss: {float(train_loss):.4f}\n")
        f.write(f"Test Loss: {float(test_loss):.4f}\n")
        f.write(f"Frozen Checkpoint: {checkpoint_path_frozen}\n")
        f.write(f"Fine-tuned Checkpoint: {checkpoint_path_fine}\n")
        f.write(f"Final Weights: {final_model_path}\n")
    
    logger.info(f"Metrics saved: {metrics_path}")
    
    # ========== Memory Management ==========
    if clear_session:
        logger.info("Clearing Keras session...")
        K.clear_session()
    
    if force_gc:
        logger.info("Running garbage collection...")
        gc.collect()
    
    logger.info(f"Run {run_idx} completed successfully")
    logger.info("=" * 80)
    
    return {
        'run_idx': run_idx,
        'backbone': backbone,
        'history_frozen': hist_frozen.history,
        'history_fine': hist_fine.history,
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'confusion_matrix': cm,
        'num_train_samples': len(train_samples),
        'num_test_samples': len(test_samples),
        'checkpoint_frozen': checkpoint_path_frozen,
        'checkpoint_fine': checkpoint_path_fine,
        'final_weights': final_model_path,
    }


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def train_experiment_1(
    dataset: List[Tuple],
    backbones: List[str],
    config_path: str,
    num_runs: int = 3,
) -> Dict[str, List[Dict]]:
    """
    Run Experiment 1 (2-phase training with validation monitoring).
    
    Args:
        dataset (List[Tuple]): Full dataset
        backbones (List[str]): List of backbone architectures to train.
        config_path (str): Path to the YAML configuration file.
        num_runs (int): Number of runs per backbone
        
    Returns:
        Dict[str, List[Dict]]: Results for each backbone
    """
    # Load configuration from the specified YAML file
    config = load_config(config_path)
    
    # Extract relevant config values
    base_results_dir = config['results']['base_dir']
    test_ratio = config['dataset']['test_ratio']
    
    all_backbone_results = {}
    
    for backbone in backbones:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRAINING BACKBONE: {backbone.upper()}")
        logger.info(f"{'=' * 80}\n")
        
        # Setup results folder for this backbone
        results_folder, _ = setup_results_folder_for_backbone(backbone, base_results_dir)
        
        # Save config
        save_backbone_config(backbone, config, results_folder)
        
        # Run multiple training runs
        backbone_results = []
        for run_idx in range(num_runs):
            result = train_one_run(
                run_idx=run_idx,
                dataset=dataset,
                config=config,
                results_folder=results_folder,
                backbone=backbone,
            )
            backbone_results.append(result)
        
        # Save summary
        save_experiment_summary(backbone_results, results_folder, 0, test_ratio)
        
        all_backbone_results[backbone] = backbone_results
        
        logger.info(f"\nCompleted {backbone}: {len(backbone_results)} runs\n")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("EXPERIMENT 1 COMPLETED")
    logger.info(f"{'=' * 80}\n")
    
    return all_backbone_results
