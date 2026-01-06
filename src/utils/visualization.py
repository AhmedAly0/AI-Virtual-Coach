"""
Visualization utilities for training results and model comparisons.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Sequence
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict, title_prefix: str = '') -> None:
    """Plot training and validation curves for loss and accuracy.

    Args:
        history (Dict): Training history dictionary with 'loss', 'accuracy', and optionally
            'val_loss' and 'val_accuracy' keys.
        title_prefix (str): Optional prefix for plot titles (e.g., 'Pose MLP - ').
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title_prefix}Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title_prefix}Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Training stopped at epoch: {len(history['loss'])}")
    if 'val_accuracy' in history:
        best_epoch = np.argmax(history['val_accuracy']) + 1
        best_val_acc = max(history['val_accuracy'])
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


def sort_labels_by_numeric_prefix(labels: Sequence[str]) -> List[str]:
    """Sort labels using the leading numeric prefix when available.

    Args:
        labels (Sequence[str]): Label strings such as "1) Dumbbell ...".

    Returns:
        List[str]: Labels ordered by their numeric prefix (fallbacks keep original order).
    """

    def _key(label: str):
        match = re.match(r"^(\d+)", label.strip())
        if match:
            return int(match.group(1))
        return float('inf')

    return sorted(labels, key=_key)


def save_training_curves(
    history: tf.keras.callbacks.History,
    plots_dir: str,
    run_idx: int
) -> None:
    """
    Save training curves for loss and accuracy (with validation).
    
    Args:
        history: Keras History object
        plots_dir (str): Directory to save plots
        run_idx (int): Run index for filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'training_curves_run_{run_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved: {save_path}")


def save_training_curves_train_only(
    history: tf.keras.callbacks.History,
    plots_dir: str,
    run_idx: int
) -> None:
    """
    Save training curves for loss and accuracy (training only, no validation).
    
    Args:
        history: Keras History object
        plots_dir (str): Directory to save plots
        run_idx (int): Run index for filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'training_curves_run_{run_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved: {save_path}")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    plots_dir: str,
    run_idx: int,
    normalize: bool = True
) -> None:
    """
    Save confusion matrix plot.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): Class name labels
        plots_dir (str): Directory to save plot
        run_idx (int): Run index
        normalize (bool): Whether to normalize matrix
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f'Normalized Confusion Matrix (Run {run_idx})'
    else:
        fmt = 'd'
        title = f'Confusion Matrix (Run {run_idx})'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, f'confusion_matrix_run_{run_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {save_path}")


def summarize_fold_validation_metrics(fold_metrics: List[Dict]) -> pd.DataFrame:
    """Create a dataframe + line plot summarizing per-fold validation metrics.

    Args:
        fold_metrics (List[Dict]): Entries with at least fold_idx, val_accuracy, val_macro_f1.

    Returns:
        pd.DataFrame: Tabular view for downstream display/analysis.
    """

    if not fold_metrics:
        raise ValueError("fold_metrics cannot be empty")

    fold_df = pd.DataFrame([
        {
            'fold_idx': fm.get('fold_idx'),
            'val_accuracy': fm.get('val_accuracy'),
            'val_macro_f1': fm.get('val_macro_f1'),
            'best_epoch': fm.get('best_epoch'),
        }
        for fm in fold_metrics
    ])

    plt.figure(figsize=(8, 4))
    melted = fold_df.melt(id_vars='fold_idx', value_vars=['val_accuracy', 'val_macro_f1'])
    sns.lineplot(data=melted, x='fold_idx', y='value', hue='variable', marker='o')
    plt.title('Validation accuracy vs. macro-F1 (per fold)')
    plt.ylabel('Score')
    plt.xlabel('Fold index')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fold_df


def plot_confusion_matrix_from_metrics(
    confusion: Sequence[Sequence[float]],
    current_class_order: Sequence[str],
    desired_class_order: Optional[Sequence[str]] = None,
    *,
    normalize: bool = False,
    title: str = 'Held-out test confusion matrix'
) -> None:
    """Plot a confusion matrix that may need reordering before visualization.

    Args:
        confusion (Sequence[Sequence[float]]): Matrix saved from metrics.json.
        current_class_order (Sequence[str]): Order used when matrix was generated.
        desired_class_order (Optional[Sequence[str]]): Target order; defaults to current order.
        normalize (bool): Whether to normalize rows to probabilities.
        title (str): Plot title.
    """

    cm = np.array(confusion, dtype=float)
    if cm.size == 0:
        raise ValueError("Confusion matrix is empty")

    source_order = list(current_class_order)
    target_order = list(desired_class_order) if desired_class_order else source_order

    index_map = {label: idx for idx, label in enumerate(source_order)}
    try:
        order_idx = [index_map[lbl] for lbl in target_order]
    except KeyError as exc:
        raise ValueError(f"Label {exc} in desired_class_order not found in current_class_order")

    cm = cm[np.ix_(order_idx, order_idx)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
        fmt = '.2f'
    else:
        fmt = '.0f'
        cm = cm.astype(int)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=target_order,
        yticklabels=target_order,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_per_class_f1_scores(
    per_class_f1: Dict,
    idx_to_label: Dict[int, str],
    desired_class_order: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None
) -> pd.Series:
    """Plot per-class F1 scores with optional label reordering.

    Args:
        per_class_f1 (Dict): Mapping of class indices (int/str) to F1 scores.
        idx_to_label (Dict[int, str]): Decoder from numeric id to label string.
        desired_class_order (Optional[Sequence[str]]): If provided, bars follow this order.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        pd.Series: Series of F1 scores keyed by resolved label names.
    """

    if not per_class_f1:
        raise ValueError("per_class_f1 cannot be empty")

    series = pd.Series(per_class_f1, dtype=float)

    def _map_label(key):
        if isinstance(key, (int, np.integer)):
            return idx_to_label.get(int(key), str(key))
        if isinstance(key, str) and key.isdigit():
            return idx_to_label.get(int(key), key)
        return key

    series.index = [_map_label(idx) for idx in series.index]
    series = series.astype(float)

    if desired_class_order:
        # Use reindex to follow custom ordering while keeping existing scores.
        series = series.reindex(desired_class_order)

    # Use provided axes or create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    sns.barplot(x=series.index, y=series.values, ax=ax)
    
    # Add value labels on top of each bar for precise comparison
    for i, (idx, value) in enumerate(series.items()):
        if not np.isnan(value):
            ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.08)  # Extra space at top to prevent overlap with title
    ax.set_ylabel('F1 score')
    if show_plot:
        ax.set_title('Per-class F1 (held-out test set)', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if show_plot:
        plt.tight_layout()
        plt.show()

    return series


def create_comprehensive_comparison(
    all_backbone_results: Dict[str, Dict],
    model_params_df: pd.DataFrame,
    output_dir: str
) -> str:
    """
    Create comprehensive comparison table of all backbones.
    
    Args:
        all_backbone_results (Dict): Results from load_backbone_results_with_config
        model_params_df (pd.DataFrame): Model parameter statistics
        output_dir (str): Directory to save comparison
        
    Returns:
        str: Path to saved CSV file
    """
    logger.info("Creating comprehensive comparison...")
    
    comparison_data = []
    
    for backbone, results in all_backbone_results.items():
        # Extract config for this backbone
        config = results.get('config', {})
        
        comparison_data.append({
            'backbone': backbone,
            'mean_test_acc': results['mean_test_acc'],
            'std_test_acc': results['std_test_acc'],
            'mean_train_acc': results['mean_train_acc'],
            'std_train_acc': results['std_train_acc'],
            'mean_epochs': results['mean_epochs'],
            'num_runs': results['num_runs'],
            'batch_size': config.get('batch_size', 'N/A'),
            'learning_rate': config.get('learning_rate', 'N/A'),
            'dropout': config.get('dropout', 'N/A'),
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('mean_test_acc', ascending=False)
    
    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comprehensive_comparison.csv')
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Comparison saved: {csv_path}")
    return csv_path


def generate_statistical_comparison(
    all_backbone_results: Dict[str, Dict],
    output_dir: str
) -> str:
    """
    Generate statistical comparison text report.
    
    Args:
        all_backbone_results (Dict): Results from load_backbone_results_with_config
        output_dir (str): Directory to save report
        
    Returns:
        str: Path to saved text file
    """
    logger.info("Generating statistical comparison report...")
    
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'statistical_comparison.txt')
    
    # Sort by mean test accuracy
    sorted_backbones = sorted(
        all_backbone_results.items(),
        key=lambda x: x[1]['mean_test_acc'],
        reverse=True
    )
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL COMPARISON OF BACKBONE ARCHITECTURES\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, (backbone, results) in enumerate(sorted_backbones, 1):
            f.write(f"{idx}. {backbone}\n")
            f.write(f"   Mean Test Accuracy: {results['mean_test_acc']:.4f} ± {results['std_test_acc']:.4f}\n")
            f.write(f"   Mean Train Accuracy: {results['mean_train_acc']:.4f} ± {results['std_train_acc']:.4f}\n")
            f.write(f"   Mean Epochs: {results['mean_epochs']:.1f}\n")
            f.write(f"   Number of Runs: {results['num_runs']}\n")
            f.write("\n")
        
        # Add winner
        best_backbone = sorted_backbones[0][0]
        best_acc = sorted_backbones[0][1]['mean_test_acc']
        f.write("=" * 80 + "\n")
        f.write(f"WINNER: {best_backbone} with mean test accuracy {best_acc:.4f}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Statistical comparison saved: {txt_path}")
    return txt_path


def plot_backbone_comparison(
    all_backbone_results: Dict[str, Dict],
    output_dir: str,
    metric: str = 'mean_test_acc'
) -> str:
    """
    Create bar plot comparing backbone performance.
    
    Args:
        all_backbone_results (Dict): Results from load_backbone_results_with_config
        output_dir (str): Directory to save plot
        metric (str): Metric to plot ('mean_test_acc', 'mean_train_acc')
        
    Returns:
        str: Path to saved plot
    """
    logger.info(f"Creating backbone comparison plot for {metric}...")
    
    # Sort by metric
    sorted_backbones = sorted(
        all_backbone_results.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )
    
    backbones = [b[0] for b in sorted_backbones]
    values = [b[1][metric] for b in sorted_backbones]
    errors = [b[1]['std_test_acc'] for b in sorted_backbones] if metric == 'mean_test_acc' else None
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(backbones)), values, yerr=errors if errors else None, capsize=5)
    
    # Color code
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(backbones)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Backbone Architecture')
    plt.ylabel('Accuracy')
    plt.title(f'Backbone Comparison: {metric.replace("_", " ").title()}')
    plt.xticks(range(len(backbones)), backbones, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'backbone_comparison_{metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plot saved: {plot_path}")
    return plot_path


def display_multi_run_summary(
    multi_run_results: List[Dict],
    aggregated_stats: Dict
) -> pd.DataFrame:
    """
    Display aggregated statistics from multi-run experiment.
    
    Args:
        multi_run_results (List[Dict]): List of individual run results
        aggregated_stats (Dict): Aggregated statistics across all runs
        
    Returns:
        pd.DataFrame: DataFrame with all run results sorted by test accuracy
    """
    print("AGGREGATED STATISTICS (30 runs)")
    print("="*60)
    print(f"\nTest Accuracy:  {aggregated_stats['test_accuracy']['mean']:.4f} ± {aggregated_stats['test_accuracy']['std']:.4f}")
    print(f"  Min: {aggregated_stats['test_accuracy']['min']:.4f}")
    print(f"  Max: {aggregated_stats['test_accuracy']['max']:.4f}")

    print(f"\nTest Macro F1:  {aggregated_stats['test_macro_f1']['mean']:.4f} ± {aggregated_stats['test_macro_f1']['std']:.4f}")
    print(f"  Min: {aggregated_stats['test_macro_f1']['min']:.4f}")
    print(f"  Max: {aggregated_stats['test_macro_f1']['max']:.4f}")

    # Create DataFrame with all run results
    runs_data = []
    for r in multi_run_results:
        runs_data.append({
            'run_idx': r['run_idx'],
            'seed': r['seed'],
            'test_accuracy': r['test_metrics']['accuracy'],
            'test_macro_f1': r['test_metrics']['macro_f1']
        })
    runs_df = pd.DataFrame(runs_data)
    runs_df = runs_df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)

    # Summary statistics for per-class F1
    print("\n\nPer-Class F1 Statistics (Mean ± Std):")
    if 'per_class_f1' in aggregated_stats:
        per_class_df = pd.DataFrame(aggregated_stats['per_class_f1']).T
        per_class_df.index.name = 'Class'
        per_class_df = per_class_df.round(4)
        print(per_class_df.to_string())
    
    return runs_df


def plot_multi_run_distributions(
    multi_run_results: List[Dict],
    aggregated_stats: Dict
) -> None:
    """
    Plot distributions of test accuracy and F1 across multiple runs.
    
    Args:
        multi_run_results (List[Dict]): List of individual run results
        aggregated_stats (Dict): Aggregated statistics across all runs
    """
    test_accs = [r['test_metrics']['accuracy'] for r in multi_run_results]
    test_f1s = [r['test_metrics']['macro_f1'] for r in multi_run_results]
    
    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(test_accs, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(aggregated_stats['test_accuracy']['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].set_xlabel('Test Accuracy', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Test Accuracy Histogram', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(test_f1s, bins=15, alpha=0.7, color='seagreen', edgecolor='black')
    axes[1].axvline(aggregated_stats['test_macro_f1']['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Test Macro F1', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Test Macro F1 Histogram', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_worst_comparison(
    multi_run_results: List[Dict],
    label_names: List[str]
) -> pd.DataFrame:
    """
    Compare best vs worst runs with confusion matrices and per-class F1.
    
    Args:
        multi_run_results (List[Dict]): List of individual run results
        label_names (List[str]): Class label names
        
    Returns:
        pd.DataFrame: Comparison of per-class F1 scores
    """
    best_run = max(multi_run_results, key=lambda x: x['test_metrics']['accuracy'])
    worst_run = min(multi_run_results, key=lambda x: x['test_metrics']['accuracy'])

    print(f"Best Run:  #{best_run['run_idx']} (seed={best_run['seed']}) - Acc: {best_run['test_metrics']['accuracy']:.4f}, F1: {best_run['test_metrics']['macro_f1']:.4f}")
    print(f"Worst Run: #{worst_run['run_idx']} (seed={worst_run['seed']}) - Acc: {worst_run['test_metrics']['accuracy']:.4f}, F1: {worst_run['test_metrics']['macro_f1']:.4f}")

    # Plot confusion matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, run, title_prefix in zip(axes, [best_run, worst_run], ['Best', 'Worst']):
        cm = np.array(run['test_metrics']['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'{title_prefix} Run (#{run["run_idx"]}, seed={run["seed"]})\n'
                     f'Acc: {run["test_metrics"]["accuracy"]:.4f} | F1: {run["test_metrics"]["macro_f1"]:.4f}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        
        if label_names:
            ordered_labels = sort_labels_by_numeric_prefix(label_names)
            short_labels = [label.split(')')[0] + ')' if ')' in label else label[:15] for label in ordered_labels]
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(short_labels, rotation=0, fontsize=9)

    plt.tight_layout()
    plt.show()

    # Per-class F1 comparison
    print("\nPer-Class F1 Comparison (Best vs Worst):")
    comparison_df = pd.DataFrame({
        'Best Run': best_run['test_metrics']['per_class_f1'],
        'Worst Run': worst_run['test_metrics']['per_class_f1']
    })
    comparison_df['Difference'] = comparison_df['Best Run'] - comparison_df['Worst Run']
    comparison_df = comparison_df.round(4)
    comparison_df.index.name = 'Class'
    print(comparison_df.to_string())
    
    return comparison_df


def plot_dual_training_history(
    history_1: Dict,
    history_2: Dict,
    model_names: List[str] = ['Model 1', 'Model 2']
) -> None:
    """Plot training curves for two models side-by-side in a 2x2 layout.
    
    Args:
        history_1 (Dict): Training history for first model (keys: loss, accuracy, val_loss, val_accuracy).
        history_2 (Dict): Training history for second model (same structure).
        model_names (List[str]): Names for the two models (for titles).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Model 1 - Loss
    axes[0, 0].plot(history_1['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history_1:
        axes[0, 0].plot(history_1['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{model_names[0]}: Loss Curves', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Model 1 - Accuracy
    axes[1, 0].plot(history_1['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history_1:
        axes[1, 0].plot(history_1['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title(f'{model_names[0]}: Accuracy Curves', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model 2 - Loss
    axes[0, 1].plot(history_2['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history_2:
        axes[0, 1].plot(history_2['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'{model_names[1]}: Loss Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model 2 - Accuracy
    axes[1, 1].plot(history_2['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history_2:
        axes[1, 1].plot(history_2['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title(f'{model_names[1]}: Accuracy Curves', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"{model_names[0]} training: {len(history_1['loss'])} epochs")
    print(f"{model_names[1]} training: {len(history_2['loss'])} epochs")


def compare_multi_run_stats(
    stats_1: Dict,
    stats_2: Dict,
    model_names: List[str] = ['Model 1', 'Model 2']
) -> pd.DataFrame:
    """Compare aggregated statistics from two multi-run experiments.
    
    Args:
        stats_1 (Dict): Aggregated stats from first multi-run experiment.
        stats_2 (Dict): Aggregated stats from second multi-run experiment.
        model_names (List[str]): Names for the two models.
        
    Returns:
        pd.DataFrame: Comparison table with mean ± std for key metrics.
    """
    print("=" * 80)
    print(f"MULTI-RUN STATISTICAL COMPARISON: {model_names[0]} vs {model_names[1]}")
    print("=" * 80)
    
    comparison_data = []
    
    # Test Accuracy
    comparison_data.append({
        'Metric': 'Test Accuracy',
        model_names[0]: f"{stats_1['test_accuracy']['mean']:.4f} ± {stats_1['test_accuracy']['std']:.4f}",
        model_names[1]: f"{stats_2['test_accuracy']['mean']:.4f} ± {stats_2['test_accuracy']['std']:.4f}"
    })
    
    # Test Macro F1
    comparison_data.append({
        'Metric': 'Test Macro F1',
        model_names[0]: f"{stats_1['test_macro_f1']['mean']:.4f} ± {stats_1['test_macro_f1']['std']:.4f}",
        model_names[1]: f"{stats_2['test_macro_f1']['mean']:.4f} ± {stats_2['test_macro_f1']['std']:.4f}"
    })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    print(f"\n{'=' * 80}\n")
    
    return df


def plot_aggregated_confusion_matrix(multi_run_results, label_names, desired_class_order=None, normalize=True, save_path=None):
    """
    Compute and plot the aggregated (mean) confusion matrix across multiple runs.
    
    Args:
        multi_run_results (list): List of result dictionaries from multiple runs.
        label_names (list): List of all class names.
        desired_class_order (list, optional): Desired order of class names for display. If None, uses label_names order.
        normalize (bool): If True, normalize confusion matrix rows to sum to 1 (show percentages).
        save_path (str, optional): Path to save the figure. If None, displays the figure.
    
    Returns:
        np.ndarray: Mean confusion matrix (shape: [num_classes, num_classes])
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract confusion matrices from all runs
    confusion_matrices = []
    for run_result in multi_run_results:
        if 'test_metrics' in run_result and 'confusion_matrix' in run_result['test_metrics']:
            cm = np.array(run_result['test_metrics']['confusion_matrix'])
            confusion_matrices.append(cm)
    
    if len(confusion_matrices) == 0:
        print("Error: No confusion matrices found in multi_run_results.")
        return None
    
    # Compute mean confusion matrix
    confusion_matrices_array = np.array(confusion_matrices)  # Shape: [num_runs, num_classes, num_classes]
    mean_cm = np.mean(confusion_matrices_array, axis=0)
    std_cm = np.std(confusion_matrices_array, axis=0)
    
    # Reorder if desired_class_order is provided
    if desired_class_order is not None:
        # Create a mapping from original label_names to indices
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        
        # Filter desired_class_order to only include classes that exist in the data
        valid_desired_labels = [label for label in desired_class_order if label in label_to_idx]
        
        # Check if any classes were filtered out
        missing_classes = [label for label in desired_class_order if label not in label_to_idx]
        if missing_classes:
            print(f"⚠️  Warning: The following classes in desired_class_order are not present in the data:")
            for mc in missing_classes:
                print(f"     - {mc}")
            print(f"   These will be skipped.\n")
        
        # Get reorder indices for valid classes only
        reorder_indices = [label_to_idx[label] for label in valid_desired_labels]
        
        # Reorder rows and columns
        mean_cm = mean_cm[reorder_indices, :][:, reorder_indices]
        std_cm = std_cm[reorder_indices, :][:, reorder_indices]
        display_labels = valid_desired_labels
    else:
        display_labels = label_names
    
    # Normalize if requested (row-wise normalization)
    if normalize:
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        mean_cm_normalized = mean_cm / row_sums
    else:
        mean_cm_normalized = mean_cm
    
    # Plot the aggregated confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use seaborn heatmap for better visualization
    sns.heatmap(
        mean_cm_normalized,
        annot=True,
        fmt='.2f' if normalize else '.1f',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    
    title = f'Aggregated Confusion Matrix (Mean across {len(confusion_matrices)} runs)'
    if normalize:
        title += ' - Normalized'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aggregated confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    return mean_cm
