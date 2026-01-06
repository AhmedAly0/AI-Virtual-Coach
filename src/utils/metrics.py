"""
Metrics and experiment tracking utilities.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import List, Dict, Tuple
from tensorflow.keras import backend as K

logger = logging.getLogger(__name__)


def save_experiment_summary(
    all_results: List[Dict],
    results_folder: str,
    run_index: int,
    test_ratio: float
) -> Tuple[str, str]:
    """
    Save experiment summary with statistics across multiple runs.
    
    Args:
        all_results (List[Dict]): List of result dictionaries from each run
        results_folder (str): Directory to save summary
        run_index (int): Experiment run index
        test_ratio (float): Test set ratio used
        
    Returns:
        Tuple[str, str]: Paths to (summary_text_file, summary_json_file)
    """
    if not all_results:
        logger.warning("No results to summarize")
        return None, None
    
    logger.info("Generating experiment summary...")
    
    # Extract metrics
    train_accs = [r['train_acc'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    
    summary_txt_path = os.path.join(results_folder, f"experiment_summary.txt")
    summary_json_path = os.path.join(results_folder, f"experiment_results.json")
    
    # Text summary
    with open(summary_txt_path, 'w') as f:
        f.write(f"Experiment Summary - Run {run_index}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Runs: {len(all_results)}\n")
        f.write(f"Test Ratio: {test_ratio}\n\n")
        
        f.write("Training Accuracy Statistics:\n")
        f.write(f"  Mean: {np.mean(train_accs):.4f}\n")
        f.write(f"  Std:  {np.std(train_accs):.4f}\n")
        f.write(f"  Min:  {np.min(train_accs):.4f}\n")
        f.write(f"  Max:  {np.max(train_accs):.4f}\n\n")
        
        f.write("Test Accuracy Statistics:\n")
        f.write(f"  Mean: {np.mean(test_accs):.4f}\n")
        f.write(f"  Std:  {np.std(test_accs):.4f}\n")
        f.write(f"  Min:  {np.min(test_accs):.4f}\n")
        f.write(f"  Max:  {np.max(test_accs):.4f}\n\n")
        
        f.write("Individual Run Results:\n")
        for r in all_results:
            f.write(f"  Run {r['run_idx']}: train_acc={r['train_acc']:.4f}, test_acc={r['test_acc']:.4f}\n")
    
    # JSON summary
    summary_data = {
        'run_index': run_index,
        'num_runs': len(all_results),
        'test_ratio': test_ratio,
        'train_accuracy': {
            'mean': float(np.mean(train_accs)),
            'std': float(np.std(train_accs)),
            'min': float(np.min(train_accs)),
            'max': float(np.max(train_accs)),
            'values': [float(x) for x in train_accs]
        },
        'test_accuracy': {
            'mean': float(np.mean(test_accs)),
            'std': float(np.std(test_accs)),
            'min': float(np.min(test_accs)),
            'max': float(np.max(test_accs)),
            'values': [float(x) for x in test_accs]
        },
        'individual_runs': [{
            'run_idx': r['run_idx'],
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'train_loss': r['train_loss'],
            'test_loss': r['test_loss']
        } for r in all_results]
    }
    
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Summary saved: {summary_txt_path}")
    logger.info(f"JSON saved: {summary_json_path}")
    
    return summary_txt_path, summary_json_path


def save_backbone_config(
    backbone: str,
    config: Dict,
    results_folder: str
) -> None:
    """
    Save training configuration for a backbone run.
    
    Args:
        backbone (str): Backbone name
        config (Dict): Configuration dictionary
        results_folder (str): Results folder path
    """
    config_path = os.path.join(results_folder, 'training_config.json')
    
    config_to_save = {
        'backbone': backbone,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    logger.info(f"Config saved: {config_path}")


def get_model_parameters(backbone: str, img_size: int = 224, num_classes: int = 15) -> Dict:
    """
    Get model parameter counts WITHOUT training.
    
    Args:
        backbone (str): Backbone architecture name
        img_size (int): Input image size
        num_classes (int): Number of classes
        
    Returns:
        Dict: Parameter statistics
    """
    from ..models import build_model_for_backbone
    
    logger.info(f"Counting parameters for {backbone}...")
    
    try:
        # Build model
        model, _ = build_model_for_backbone(backbone, img_size, num_classes)
        
        # Get parameter counts
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        # Get layer count
        num_layers = len(model.layers)
        
        # Clean up
        del model
        K.clear_session()
        
        return {
            'backbone': backbone,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(non_trainable_params),
            'num_layers': num_layers,
            'params_millions': round(total_params / 1e6, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting parameters for {backbone}: {e}")
        return None


def get_all_model_parameters(
    backbones: List[str],
    img_size: int = 224,
    num_classes: int = 15
) -> pd.DataFrame:
    """
    Get parameter counts for all backbones.
    
    Args:
        backbones (List[str]): List of backbone names
        img_size (int): Input image size
        num_classes (int): Number of classes
        
    Returns:
        pd.DataFrame: DataFrame with parameter statistics
    """
    from tqdm import tqdm
    
    logger.info(f"Counting parameters for {len(backbones)} backbones...")
    
    all_params = []
    
    for bb in tqdm(backbones, desc="Counting parameters"):
        params = get_model_parameters(bb, img_size, num_classes)
        if params:
            all_params.append(params)
    
    df = pd.DataFrame(all_params)
    df = df.sort_values('total_params', ascending=True)
    
    logger.info("Parameter counting complete")
    return df


def load_backbone_results_with_config(results_base_dir: str = 'results') -> Dict[str, Dict]:
    """
    Load all experiment summaries with their training configurations.
    
    Args:
        results_base_dir (str): Base directory containing backbone results
        
    Returns:
        Dict[str, Dict]: Dictionary mapping backbone names to their results
    """
    logger.info(f"Loading backbone results from: {results_base_dir}")
    
    all_backbone_results = {}
    
    if not os.path.exists(results_base_dir):
        logger.warning(f"Results directory not found: {results_base_dir}")
        return all_backbone_results
    
    # Iterate through backbone folders
    for backbone_name in os.listdir(results_base_dir):
        backbone_path = os.path.join(results_base_dir, backbone_name)
        
        if not os.path.isdir(backbone_path):
            continue
        
        # Skip 'comparisons' folder
        if backbone_name == 'comparisons':
            continue
        
        # Find the most recent run folder
        run_folders = [d for d in os.listdir(backbone_path) if d.startswith('run_')]
        if not run_folders:
            logger.warning(f"No run folders found in {backbone_path}")
            continue
        
        # Sort and get latest
        latest_run = sorted(run_folders)[-1]
        run_path = os.path.join(backbone_path, latest_run)
        
        # Look for JSON summary file
        json_file = os.path.join(run_path, 'experiment_results.json')
        
        if not os.path.exists(json_file):
            logger.warning(f"No JSON summary found in {run_path}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            all_backbone_results[backbone_name] = {
                'mean_test_acc': data['test_accuracy']['mean'],
                'std_test_acc': data['test_accuracy']['std'],
                'min_test_acc': data['test_accuracy']['min'],
                'max_test_acc': data['test_accuracy']['max'],
                'test_acc_values': data['test_accuracy']['values'],
                'mean_train_acc': data['train_accuracy']['mean'],
                'num_runs': data['num_runs'],
                'run_path': run_path,
                'json_path': json_file
            }
            logger.info(f"Loaded {backbone_name}: mean_acc={data['test_accuracy']['mean']:.4f}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Successfully loaded {len(all_backbone_results)} backbone results")
    return all_backbone_results


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute macro F1 score for integer-encoded labels."""

    eps = 1e-8
    f1_scores = []
    for cls in range(num_classes):
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))

        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def per_class_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Return per-class F1 scores as a dictionary."""

    scores = {}
    eps = 1e-8
    for cls in range(num_classes):
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        scores[cls] = float(f1)

    return scores
