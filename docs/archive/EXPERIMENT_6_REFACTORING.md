# Experiment 6 Refactoring Summary

**Date:** December 21, 2025  
**Objective:** Add YAML-based configuration and multi-run support to Experiment 6 (Pose MLP baseline)

## Overview

Refactored [experiment_6.py](../src/scripts/experiment_6.py) to match the pattern established in Experiment 5, enabling:
- YAML-based hyperparameter configuration
- Multi-run training with different random seeds
- Automatic result aggregation with mean ± std statistics
- Organized results directory structure
- Memory management and cleanup
- Backward compatibility with legacy parameter passing

## Changes Made

### 1. Configuration Files

#### Created: `config/experiment_6.yaml`
Static pose features configuration with sections:
- **dataset:** val_ratio (0.15), test_ratio (0.30), random_seed (42), stratified splits
- **model:** hidden_sizes ([256, 128]), dropout (0.25), num_classes (15)
- **training:** batch_size (64), lr (0.001), max_epochs (80), optimizer (adam)
- **callbacks:** early_stopping with patience (10), monitor (val_loss)
- **multi_run:** enabled (true), num_runs (30), base_seed (42)
- **results:** base_dir for outputs (`.../exp_06_pose_mlp_static`)
- **memory:** clear_session_after_run, force_gc_after_run
- **metrics:** primary (macro_f1), secondary (accuracy, loss)

#### Created: `config/experiment_6_temporal.yaml`
Identical structure but with results directory: `.../exp_06_pose_mlp_temporal`  
Reserved for future temporal pose features training.

### 2. Source Code Changes

#### Modified: `src/scripts/experiment_6.py`

**New Imports:**
```python
import gc
import json
import os
from typing import Optional
from src.utils.io_utils import load_config, setup_results_folder
```

**Refactored Function: `train_experiment_6_static()`**
- **New Parameters:**
  - `config_path` (str): Path to YAML config file (default: 'config/experiment_6.yaml')
  - `results_folder` (Optional[str]): Pre-created results folder (for multi-run mode)
  - `run_idx` (Optional[int]): Run index for naming
  
- **Legacy Parameters:** All original parameters now optional, loaded from config if not provided
  - Enables backward compatibility with existing code
  
- **New Functionality:**
  - Loads hyperparameters from YAML configuration
  - Creates results directory with auto-incrementing names (`run_001`, `run_002`, etc.)
  - Saves metrics to `metrics.json` in results folder
  - Includes `run_idx` and `seed` in results dictionary
  - Configurable callbacks from YAML (early stopping parameters)

**New Helper: `_compute_aggregation_stats_exp6(all_run_results)`**
- Calculates mean, std, min, max across multiple runs for:
  - test_accuracy
  - test_macro_f1
  - per_class_f1 (for each of 15 exercise classes)
- Returns aggregated statistics dictionary

**New Helper: `_save_multi_run_summary_exp6(all_run_results, aggregated_stats, multi_run_folder)`**
- Saves three files:
  1. `aggregated_stats.json`: Machine-readable statistics
  2. `aggregated_summary.txt`: Human-readable summary with per-class breakdown
  3. `all_runs.json`: Complete results from all individual runs

**New Function: `train_experiment_6_multi_run(npz_path, config_path)`**
- Main multi-run orchestration function
- Validates `multi_run.enabled` is True in config
- Creates parent folder: `multi_run_001`, `multi_run_002`, etc.
- Loops through `num_runs` executions:
  - Uses seed = base_seed + run_number for each run
  - Creates individual run folders: `run_001/`, `run_002/`, etc.
  - Trains model with `train_experiment_6_static()`
  - Collects results from each run
  - Performs memory cleanup (clear_session, gc.collect)
- Computes aggregated statistics
- Saves comprehensive multi-run summary
- Returns: `(all_run_results, aggregated_stats)`

**Updated Exports:**
```python
__all__ = ['train_experiment_6_static', 'train_experiment_6_multi_run']
```

### 3. Documentation & Testing

#### Created: `tests/test_experiment_6.py`
Validation script with three test cases:
1. **Single run with config:** Verifies YAML loading and results saving
2. **Multi-run (3 runs):** Quick validation of multi-run infrastructure
3. **Backward compatibility:** Ensures legacy parameter passing still works

To run tests:
```bash
cd /mnt/d/Graduation_Project/ai-virtual-coach
python tests/test_experiment_6.py
```

#### Updated: `notebooks/exer_recog/06_pose_mlp.ipynb`
- Added new markdown cell documenting config-based approach
- Provided usage examples for both single-run and multi-run modes
- Maintained backward compatibility with existing notebook cells

## Directory Structure

### Single Run Output
```
experiments/exer_recog/results/exp_06_pose_mlp_static/
├── run_001/
│   └── metrics.json
├── run_002/
│   └── metrics.json
└── ...
```

### Multi-Run Output
```
experiments/exer_recog/results/
├── multi_run_001/
│   ├── config.yaml                 # Copy of configuration used
│   ├── aggregated_stats.json       # Statistical summary
│   ├── aggregated_summary.txt      # Human-readable report
│   ├── all_runs.json              # All individual results
│   ├── run_001/
│   │   └── metrics.json
│   ├── run_002/
│   │   └── metrics.json
│   └── ...
└── multi_run_002/
    └── ...
```

## Usage Examples

### Single Run (Config-Based)
```python
from src.scripts.experiment_6 import train_experiment_6_static

results = train_experiment_6_static(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)

print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test Macro F1: {results['test_metrics']['macro_f1']:.4f}")
```

### Multi-Run (30 Runs with Statistical Aggregation)
```python
from src.scripts.experiment_6 import train_experiment_6_multi_run

all_runs, stats = train_experiment_6_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)

print(f"Accuracy: {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
print(f"Macro F1: {stats['test_macro_f1']['mean']:.4f} ± {stats['test_macro_f1']['std']:.4f}")
```

### Backward Compatibility (Legacy Style)
```python
# Still works - parameters override config values
results = train_experiment_6_static(
    npz_path='datasets/Mediapipe pose estimates/pose_data_side_static.npz',
    config_path='config/experiment_6.yaml',
    seed=99,
    batch_size=32,
    hidden_sizes=(512, 256, 128),
    max_epochs=100
)
```

## Key Benefits

1. **Reproducibility:** All hyperparameters stored in version-controlled YAML files
2. **Statistical Rigor:** Multi-run support eliminates single-run variance, provides confidence intervals
3. **Organized Results:** Auto-incrementing folders prevent overwrites, maintain experiment history
4. **Consistency:** Matches Experiment 5 patterns for easier codebase navigation
5. **Flexibility:** Backward compatibility ensures existing code/notebooks continue working
6. **Memory Efficiency:** Automatic cleanup prevents OOM errors in long multi-run experiments
7. **Future-Ready:** Temporal config prepared for future temporal pose features

## Future Work

1. **Temporal Training Function:** Create `train_experiment_6_temporal()` using temporal pose features
2. **Fusion Experiments:** Combine front + side views, or static + temporal features
3. **Hyperparameter Tuning:** Add grid search/random search capabilities
4. **Visualization:** Auto-generate comparison plots across multi-runs
5. **Early Stopping Analysis:** Track which epochs models converged at across runs

## Configuration Tips

### Adjust Number of Runs
```yaml
multi_run:
  num_runs: 10  # Reduce for faster experiments
```

### Change Results Directory
```yaml
results:
  base_dir: /path/to/custom/results/folder
```

### Modify Model Architecture
```yaml
model:
  hidden_sizes: [512, 256, 128]  # Deeper network
  dropout: 0.35                   # Higher regularization
```

### Tune Training
```yaml
training:
  batch_size: 128     # Larger batches
  lr: 0.0005         # Lower learning rate
  max_epochs: 120    # Longer training
```

---

**Implementation Status:** ✅ Complete  
**Tested:** ✅ Syntax validated, ready for functional testing  
**Documentation:** ✅ Complete with examples and test scripts
