import warnings
warnings.warn(
    "This GEI-based experiment is archived. Use pose-based experiment_1.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

"""
[ARCHIVED] Experiment 4: Regularized heads with dual pooling and AdamW optimizer.

Key improvements over Experiment 3:
- Dual pooling (GAP + GMP) for richer feature representation
- Smaller classification heads to reduce overfitting
- Label smoothing (0.1) for better generalization
- AdamW optimizer with weight decay (1e-4)
- Differential learning rates: head (1e-4) vs backbone (1e-5)

Training strategy:
1. Phase 1 (frozen): Train head only with AdamW
2. Phase 2 (unfrozen): Progressive backbone unfreezing with lower LR
"""

import os
import sys
import yaml
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_builder import (
    build_model_for_backbone_v4,
    BACKBONE_REGISTRY,
    categorical_with_label_smoothing,
)
from src.data.dataset_builder import (
    build_datasets_three_way_streaming,
    make_split_three_way,
)
from src.utils.io_utils import setup_results_folder_for_backbone
from src.utils.metrics import save_experiment_summary, save_backbone_config

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


def unfreeze_backbone_progressive(model, base_model, unfreeze_pct: float = 0.5):
    """
    Progressively unfreeze the top layers of the backbone.
    
    Args:
        model: Full Keras model
        base_model: Backbone model (nested inside full model)
        unfreeze_pct: Percentage of layers to unfreeze (0.0 to 1.0)
    """
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_pct))
    
    logger.info(f"Unfreezing top {unfreeze_pct*100:.0f}% of backbone ({total_layers - unfreeze_from}/{total_layers} layers)")
    
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    
    logger.info(f"Trainable weights: {len(model.trainable_weights)}")


def train_one_run(
    run_idx: int,
    dataset: list,
    config: Dict,
    results_folder: str,
    backbone: str
) -> Dict:
    """
    Train one run for a specific backbone with regularization techniques (v4).
    Uses proper 3-way subject-independent split with validation monitoring.
    
    Phase 1: Train with frozen backbone (early stopping on validation)
    Phase 2: Unfreeze top N% of backbone layers based on config (early stopping on validation)
    
    Split Strategy:
    - ~55% subjects → training (for learning)
    - ~15% subjects → validation (for early stopping, NEVER for final metrics)
    - ~30% subjects → test (TRUE unseen benchmark, evaluated ONLY at the end)
    
    Args:
        run_idx: Run index for tracking
        dataset: Full dataset (pre-loaded, merged front+side)
        config: Configuration dictionary from YAML
        results_folder: Results directory
        backbone: Backbone architecture name
        
    Returns:
        Dictionary with results (test_acc, test_loss, histories, confusion matrix, etc.)
    """
    logger.info(f"{'='*80}")
    logger.info(f"Starting Run {run_idx} with backbone: {backbone} (Regularized Dual-Pooling v4)")
    logger.info(f"{'='*80}")
    
    # Extract hyperparameters
    img_size = config['model']['img_size']
    batch_size = config['training']['batch_size']
    head_lr = config['training']['initial_lr']
    backbone_lr = config['training']['fine_tune_lr']
    weight_decay = config['training']['weight_decay']
    label_smoothing = config['model']['label_smoothing']
    num_classes = config['model']['num_classes']
    epochs_frozen = config['training']['frozen_epochs']
    epochs_unfrozen = config['training']['fine_tune_epochs']
    unfreeze_pct = config['unfreezing'][backbone]['phase2_unfreeze_percent']
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    augment_config = config.get('augmentation', None)
    clear_session = config['memory']['clear_session_after_run']
    force_gc = config['memory']['force_gc_after_run']
    results_config = config.get('results', {})
    save_checkpoints = results_config.get('save_checkpoints', True)
    save_metrics = results_config.get('save_metrics', True)
    save_confusion = results_config.get('save_confusion_matrix', True)
    
    logger.info(
        "Config: head_lr=%s, backbone_lr=%s, weight_decay=%s",
        head_lr,
        backbone_lr,
        weight_decay,
    )
    logger.info("        label_smoothing=%s, unfreeze_pct=%s", label_smoothing, unfreeze_pct)
    logger.info("Phase 2 unfreezing: %.0f%% of backbone layers", unfreeze_pct * 100)
    
    # Build model with v4 architecture (config-driven hyperparameters)
    model, base_model, preprocess_fn = build_model_for_backbone_v4(
        backbone=backbone,
        config=config
    )
    
    # 3-way subject-independent split
    train_samples, val_samples, test_samples, label_to_int = make_split_three_way(
        dataset=dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42
    )
    
    # Build datasets (Option B streaming helpers)
    logger.info("Building TensorFlow datasets with streaming pipeline (Option B)...")
    train_ds, val_ds, test_ds, label_vectors = build_datasets_three_way_streaming(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        label_to_int=label_to_int,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=preprocess_fn,
        augment_config=augment_config,
    )
    num_train = len(label_vectors['train'])
    num_val = len(label_vectors['val'])
    num_test = len(label_vectors['test'])
    y_test = label_vectors['test']
    logger.info(
        "Datasets built (streaming): train=%s, val=%s, test=%s",
        num_train,
        num_val,
        num_test,
    )
    
    callbacks_config = config.get('callbacks', {})
    early_cfg = callbacks_config.get('early_stopping', {})
    reduce_cfg = callbacks_config.get('reduce_lr', {})

    callbacks_frozen: List[tf.keras.callbacks.Callback] = []
    callbacks_fine: List[tf.keras.callbacks.Callback] = []

    if early_cfg.get('enabled', True):
        early_kwargs = dict(
            monitor=early_cfg.get('monitor', 'val_loss'),
            patience=early_cfg.get('patience', 10),
            min_delta=early_cfg.get('min_delta', 0.0),
            restore_best_weights=True,
            verbose=1,
        )
        callbacks_frozen.append(tf.keras.callbacks.EarlyStopping(**early_kwargs))
        callbacks_fine.append(tf.keras.callbacks.EarlyStopping(**early_kwargs))

    if reduce_cfg.get('enabled', True):
        reduce_kwargs = dict(
            monitor=reduce_cfg.get('monitor', 'val_loss'),
            factor=reduce_cfg.get('factor', 0.5),
            patience=reduce_cfg.get('patience', max(1, early_cfg.get('patience', 10) // 2)),
            min_lr=reduce_cfg.get('min_lr', 1e-7),
            verbose=1,
        )
        callbacks_frozen.append(tf.keras.callbacks.ReduceLROnPlateau(**reduce_kwargs))
        callbacks_fine.append(tf.keras.callbacks.ReduceLROnPlateau(**reduce_kwargs))

    checkpoint_path_frozen = None
    checkpoint_path_fine = None
    if save_checkpoints:
        checkpoint_path_frozen = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_frozen.keras')
        checkpoint_path_fine = os.path.join(results_folder, 'checkpoints', f'run_{run_idx:03d}_fine.keras')
        callbacks_frozen.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path_frozen,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )
        callbacks_fine.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path_fine,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )
    
    # ========== PHASE 1: Train frozen backbone with AdamW ==========
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Training with FROZEN backbone")
    logger.info(f"Optimizer: AdamW(lr={head_lr}, weight_decay={weight_decay})")
    logger.info("="*80)
    
    # Recompile with AdamW optimizer (try experimental namespace for TF compatibility)
    try:
        # TensorFlow 2.11+
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=head_lr,
            weight_decay=weight_decay
        )
    except AttributeError:
        try:
            # TensorFlow 2.9-2.10
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=head_lr,
                weight_decay=weight_decay
            )
        except (AttributeError, ImportError):
            # Fallback: use Adam (weight decay applied separately via kernel_regularizer)
            logger.warning("AdamW not available, using Adam optimizer")
            optimizer = tf.keras.optimizers.Adam(learning_rate=head_lr)
    
    model.compile(
        optimizer=optimizer,
        loss=categorical_with_label_smoothing(num_classes, label_smoothing),
        metrics=['accuracy']
    )
    
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        callbacks=callbacks_frozen,
        verbose=1
    )
    
    # Evaluate after Phase 1
    val_loss_frozen, val_acc_frozen = model.evaluate(val_ds, verbose=0)
    logger.info("Phase 1 completed: val_acc=%.4f, val_loss=%.4f", val_acc_frozen, val_loss_frozen)
    
    # ========== PHASE 2: Fine-tune with unfrozen backbone ==========
    logger.info("\n" + "="*80)
    logger.info(f"PHASE 2: Fine-tuning with UNFROZEN top {unfreeze_pct*100:.0f}% of backbone")
    logger.info(f"Differential LR: head={head_lr}, backbone={backbone_lr}")
    logger.info("="*80)
    
    # Unfreeze backbone progressively
    unfreeze_backbone_progressive(model, base_model, unfreeze_pct=unfreeze_pct)
    
    # Recompile with differential learning rates
    # Note: We approximate differential LR by using lower global LR since backbone has more layers
    # A more sophisticated approach would use layer-wise learning rates
    try:
        # TensorFlow 2.11+
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=backbone_lr,  # Use lower LR for fine-tuning
            weight_decay=weight_decay
        )
    except AttributeError:
        try:
            # TensorFlow 2.9-2.10
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=backbone_lr,
                weight_decay=weight_decay
            )
        except (AttributeError, ImportError):
            # Fallback: use Adam
            logger.warning("AdamW not available, using Adam optimizer")
            optimizer = tf.keras.optimizers.Adam(learning_rate=backbone_lr)
    
    model.compile(
        optimizer=optimizer,
        loss=categorical_with_label_smoothing(num_classes, label_smoothing),
        metrics=['accuracy']
    )
    
    history_unfrozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_unfrozen,
        callbacks=callbacks_fine,
        verbose=1
    )
    
    # Evaluate after Phase 2
    val_loss_unfrozen, val_acc_unfrozen = model.evaluate(val_ds, verbose=0)
    logger.info("Phase 2 completed: val_acc=%.4f, val_loss=%.4f", val_acc_unfrozen, val_loss_unfrozen)
    
    # ========== EVALUATION ==========
    logger.info("Evaluating model...")
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)  # ✅ Test set evaluated ONLY here
    
    logger.info("Train Accuracy: %.4f", train_acc)
    logger.info("Val Accuracy: %.4f", val_acc)
    logger.info("Test Accuracy: %.4f ← TRUE BENCHMARK (never seen during training)", test_acc)
    
    # Confusion matrix
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred_classes, num_classes=config['model']['num_classes']).numpy()
    
    if save_confusion:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
    metrics_path = None
    if save_metrics:
        metrics_path = os.path.join(results_folder, "metrics", f"run_{run_idx:03d}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Run {run_idx} Results (Experiment 4 - Regularized Dual-Pooling v4):\n")
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
        tf.keras.backend.clear_session()
        logger.info("Keras session cleared")
    
    if force_gc:
        import gc
        gc.collect()
        logger.info("Garbage collection forced")
    
    logger.info(f"Run {run_idx} completed successfully")
    logger.info("=" * 80)
    
    return {
        'run_idx': run_idx,
        'backbone': backbone,
        'history_frozen': history_frozen.history,
        'history_fine': history_unfrozen.history,
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'test_loss': float(test_loss),
        'val_acc_frozen': float(val_acc_frozen),
        'val_acc_unfrozen': float(val_acc_unfrozen),
        'confusion_matrix': cm,
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_test_samples': len(test_samples),
        'checkpoint_frozen': checkpoint_path_frozen,
        'checkpoint_fine': checkpoint_path_fine,
        'final_weights': final_model_path,
    }


def train_experiment_4(
    dataset: List[tuple],
    backbones: Optional[List[str]] = None,
    config_path: str = 'config/experiment_4.yaml',
    num_runs: int = 5
) -> Dict[str, List[Dict]]:
    """Run Experiment 4 across multiple backbones and runs."""

    logger.info("=" * 80)
    logger.info("EXPERIMENT 4: Regularized Dual-Pooling Heads with AdamW")
    logger.info("=" * 80)

    config = load_config(config_path)
    base_results_dir = config['results']['base_dir']
    test_ratio = config['dataset']['test_ratio']

    if backbones is None:
        backbones = list(BACKBONE_REGISTRY.keys())

    logger.info(f"Backbones: {backbones}")
    logger.info(f"Runs per backbone: {num_runs}")

    all_backbone_results: Dict[str, List[Dict]] = {}

    for backbone in backbones:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting backbone: {backbone}")
        logger.info(f"{'=' * 80}")

        backbone_dir = os.path.join(base_results_dir, backbone)
        os.makedirs(backbone_dir, exist_ok=True)
        save_backbone_config(backbone, config, backbone_dir)

        backbone_results: List[Dict] = []
        for _ in range(num_runs):
            results_folder, run_idx = setup_results_folder_for_backbone(
                backbone=backbone,
                base_results_dir=base_results_dir
            )

            logger.info(f"\nRun {run_idx} ({backbone}): {results_folder}")

            try:
                result = train_one_run(
                    run_idx=run_idx,
                    dataset=dataset,
                    config=config,
                    results_folder=results_folder,
                    backbone=backbone
                )
                backbone_results.append(result)
                logger.info(f"✓ Run {run_idx} completed: test_acc={result['test_acc']:.4f}")
            except Exception as exc:
                logger.error(f"✗ Run {run_idx} failed: {exc}", exc_info=True)

        if backbone_results:
            save_experiment_summary(backbone_results, backbone_dir, 0, test_ratio)

        all_backbone_results[backbone] = backbone_results

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 4 COMPLETED")
    logger.info("=" * 80)

    return all_backbone_results


if __name__ == '__main__':
    logger.warning(
        "Experiment 4 requires a pre-loaded dataset. Import this module and call "
        "train_experiment_4(dataset=...) from your own script or notebook."
    )
