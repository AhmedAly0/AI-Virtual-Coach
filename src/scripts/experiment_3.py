"""
Experiment 3: 2-Phase Transfer Learning with Architecture-Specific Heads

Training Strategy:
- Phase 1: Train with frozen backbone (validation-monitored early stopping)
- Phase 2: Unfreeze entire model and fine-tune

Key Features:
- Refined architecture-specific classification heads (v3)
- 2-phase training (same as Experiment 1)
- Validation monitoring with callbacks
- Basic augmentation (flip + translation)

Differences from Experiment 1:
- Uses architecture-specific heads instead of universal head


Differences from Experiment 2:
- Uses simpler 2-layer heads (not complex 3-layer)
- All backbones use GlobalAveragePooling2D
- Validation monitoring enabled (not blind training)
- 2-phase unfreezing (not 3-stage progressive)
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

from ..data import make_split_three_way, build_datasets_three_way
from ..models.model_builder import build_model_for_backbone_v3
from ..models import get_callbacks
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
    results_folder: str,
    backbone: str
) -> Dict:
    """
    Train a single run with 2-phase training using architecture-specific heads (v3).
    Uses proper 3-way subject-independent split with validation monitoring.
    
    Phase 1: Train with frozen backbone (early stopping on validation)
    Phase 2: Unfreeze top N% of backbone layers based on config (early stopping on validation)
    
    Split Strategy:
    - ~55% subjects → training (for learning)
    - ~15% subjects → validation (for early stopping, NEVER for final metrics)
    - ~30% subjects → test (TRUE unseen benchmark, evaluated ONLY at the end)
    
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
    logger.info(f"Starting Run {run_idx} with backbone: {backbone} (Architecture-Specific Head v3)")
    logger.info(f"=" * 80)
    
    # Extract parameters from config
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    img_size = config['model']['img_size']
    num_classes = config['model']['num_classes']
    batch_size = config['training']['batch_size']
    frozen_epochs = config['training']['frozen_epochs']
    fine_tune_epochs = config['training']['fine_tune_epochs']
    initial_lr = config['training']['initial_lr']
    fine_tune_lr = config['training']['fine_tune_lr']
    augment_config = config['augmentation']
    clear_session = config['memory']['clear_session_after_run']
    force_gc = config['memory']['force_gc_after_run']
    
    # Get backbone-specific unfreezing percentage
    unfreeze_percent = config['unfreezing'][backbone]['phase2_unfreeze_percent']
    logger.info(f"Phase 2 unfreezing: {unfreeze_percent*100:.0f}% of backbone layers")
    
    # 3-way subject-independent split
    train_samples, val_samples, test_samples, label_to_int = make_split_three_way(
        dataset, val_ratio, test_ratio, seed=42
    )
    
    # Build model with refined architecture-specific head (v3)
    model, preprocess_fn = build_model_for_backbone_v3(backbone, img_size, num_classes)
    
    # Build 3 datasets
    train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test = build_datasets_three_way(
        train_samples, val_samples, test_samples,
        label_to_int, batch_size, img_size, preprocess_fn, augment_config
    )
    
    # ========== PHASE 1: Frozen Training ==========
    logger.info("Phase 1: Training with frozen backbone...")
    
    checkpoint_path_frozen = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_frozen.keras')
    callbacks_frozen = get_callbacks(checkpoint_path_frozen)
    
    hist_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=frozen_epochs,
        callbacks=callbacks_frozen,
        verbose=1
    )
    
    logger.info("Phase 1 completed")
    
    # ========== PHASE 2: Progressive Fine-tuning ==========
    logger.info(f"Phase 2: Fine-tuning with {unfreeze_percent*100:.0f}% backbone unfreezing...")
    
    # Identify the base_model (backbone) within the full model
    # The model structure is: Input -> base_model (frozen backbone) -> custom head layers
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:  # Backbone has many layers
            base_model = layer
            break
    
    if base_model is None:
        logger.warning("Could not identify base_model, unfreezing all layers")
        for layer in model.layers:
            layer.trainable = True
    else:
        # Smart unfreezing: unfreeze only top N% of backbone layers
        total_backbone_layers = len(base_model.layers)
        layers_to_unfreeze = int(unfreeze_percent * total_backbone_layers)
        
        if unfreeze_percent >= 1.0:
            # Full unfreezing (100%)
            logger.info(f"  Unfreezing ALL {total_backbone_layers} backbone layers (100%)")
            base_model.trainable = True
        else:
            # Progressive unfreezing (top N%)
            logger.info(f"  Total backbone layers: {total_backbone_layers}")
            logger.info(f"  Unfreezing top {layers_to_unfreeze} layers ({unfreeze_percent*100:.0f}%)")
            logger.info(f"  Keeping bottom {total_backbone_layers - layers_to_unfreeze} layers frozen")
            
            base_model.trainable = True
            # Freeze bottom (100 - N)% layers
            for layer in base_model.layers[:-layers_to_unfreeze]:
                layer.trainable = False
            # Unfreeze top N% layers
            for layer in base_model.layers[-layers_to_unfreeze:]:
                layer.trainable = True
    
    # Count trainable parameters
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_count = sum([tf.size(w).numpy() for w in model.weights])
    logger.info(f"  Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_path_fine = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_fine.keras')
    callbacks_fine = get_callbacks(checkpoint_path_fine)
    
    hist_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks_fine,
        verbose=1
    )
    
    logger.info("Phase 2 completed")
    
    # ========== Evaluation ==========
    logger.info("Evaluating model...")
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)  # ✅ Test set evaluated ONLY here
    
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Val Accuracy: {val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f} ← TRUE BENCHMARK (never seen during training)")
    
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
        f.write(f"Run {run_idx} Results (Experiment 3 - Architecture-Specific Heads v3):\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Train Accuracy: {float(train_acc):.4f}\n")
        f.write(f"Val Accuracy: {float(val_acc):.4f}\n")
        f.write(f"Test Accuracy: {float(test_acc):.4f} (TRUE BENCHMARK)\n")
        f.write(f"Train Loss: {float(train_loss):.4f}\n")
        f.write(f"Val Loss: {float(val_loss):.4f}\n")
        f.write(f"Test Loss: {float(test_loss):.4f}\n")
        f.write(f"Frozen Checkpoint: {checkpoint_path_frozen}\n")
        f.write(f"Fine-tuned Checkpoint: {checkpoint_path_fine}\n")
        f.write(f"Final Weights: {final_model_path}\n")
    
    logger.info(f"Metrics saved: {metrics_path}")
    
    # ========== Memory Management ==========
    if clear_session:
        K.clear_session()
        logger.info("Keras session cleared")
    
    if force_gc:
        gc.collect()
        logger.info("Garbage collection forced")
    
    logger.info(f"Run {run_idx} completed successfully")
    logger.info("=" * 80)
    
    return {
        'run_idx': run_idx,
        'backbone': backbone,
        'history_frozen': hist_frozen.history,
        'history_fine': hist_fine.history,
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'test_loss': float(test_loss),
        'confusion_matrix': cm,
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_test_samples': len(test_samples),
        'checkpoint_frozen': checkpoint_path_frozen,
        'checkpoint_fine': checkpoint_path_fine,
        'final_weights': final_model_path,
    }


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def train_experiment_3(
    dataset: List[Tuple],
    backbones: List[str],
    config_path: str,
    num_runs: int = 5
) -> Dict[str, List[Dict]]:
    """
    Run Experiment 3 (2-phase training with architecture-specific heads v3 and smart unfreezing).
    
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
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    
    logger.info("=" * 80)
    logger.info("Starting Experiment 3: Architecture-Specific Heads v3 with Smart Unfreezing")
    logger.info("=" * 80)
    
    all_backbone_results = {}
    
    for backbone in backbones:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRAINING BACKBONE: {backbone.upper()} (Architecture-Specific Head v3)")
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
                backbone=backbone
            )
            backbone_results.append(result)
        
        # Save summary
        save_experiment_summary(backbone_results, results_folder, 0, test_ratio)
        
        all_backbone_results[backbone] = backbone_results
        
        logger.info(f"\nCompleted {backbone}: {len(backbone_results)} runs\n")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("EXPERIMENT 3 COMPLETED")
    logger.info(f"{'=' * 80}\n")
    
    return all_backbone_results
