"""
I/O utilities for managing experiment results and random seeds.
"""

import os
import random
import numpy as np
import tensorflow as tf
import yaml
import logging
from datetime import datetime
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global random seed set to: {seed}")


def setup_results_folder(base_results_dir: str = 'results') -> Tuple[str, int]:
    """
    Create an indexed results folder without overwriting existing runs.
    
    Args:
        base_results_dir (str): Base directory for results
        
    Returns:
        Tuple[str, int]: (results_folder_path, run_index)
    """
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Find next available run index
    run_idx = 1
    while True:
        results_folder = os.path.join(base_results_dir, f'run_{run_idx:03d}')
        if not os.path.exists(results_folder):
            break
        run_idx += 1
    
    # Create subdirectories
    for subfolder in ['models', 'plots', 'metrics', 'checkpoints']:
        os.makedirs(os.path.join(results_folder, subfolder), exist_ok=True)
    
    logger.info(f"Results folder created: {results_folder}")
    return results_folder, run_idx


def setup_results_folder_for_backbone(
    backbone: str, 
    base_results_dir: str = 'results',
    experiment_name: str = None
) -> Tuple[str, int]:
    """
    Create results folder for specific backbone without overwriting existing runs.
    
    Args:
        backbone (str): Name of backbone architecture
        base_results_dir (str): Base directory for results
        experiment_name (str, optional): Experiment name to use as subdirectory
        
    Returns:
        Tuple[str, int]: (results_folder_path, run_index)
    """
    # If experiment_name is provided, use it as part of the path
    if experiment_name:
        base_results_dir = os.path.join(base_results_dir, experiment_name)
    
    backbone_dir = os.path.join(base_results_dir, backbone)
    os.makedirs(backbone_dir, exist_ok=True)
    
    # Find next available run index for this backbone
    run_idx = 1
    while True:
        results_folder = os.path.join(backbone_dir, f'run_{run_idx:03d}')
        if not os.path.exists(results_folder):
            break
        run_idx += 1
    
    # Create subdirectories
    for subfolder in ['models', 'plots', 'metrics', 'checkpoints']:
        os.makedirs(os.path.join(results_folder, subfolder), exist_ok=True)
    
    logger.info(f"Backbone results folder: {results_folder}")
    return results_folder, run_idx


def display_results_structure(results_folder: str) -> None:
    """
    Display the structure of saved results.
    
    Args:
        results_folder (str): Results directory path
    """
    logger.info(f"\nResults structure for: {results_folder}")
    logger.info("=" * 80)
    
    for root, dirs, files in os.walk(results_folder):
        level = root.replace(results_folder, '').count(os.sep)
        indent = ' ' * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            logger.info(f"{sub_indent}{file}")
        if len(files) > 5:
            logger.info(f"{sub_indent}... and {len(files) - 5} more files")


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
    return config
