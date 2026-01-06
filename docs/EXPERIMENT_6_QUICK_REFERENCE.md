# Experiment 6 Quick Reference Guide

## 📁 Files Created/Modified

### Configuration Files
- ✅ `config/experiment_6.yaml` - Static pose features config
- ✅ `config/experiment_6_temporal.yaml` - Temporal pose features config (for future use)

### Source Code
- ✅ `src/scripts/experiment_6.py` - Refactored with YAML config and multi-run support

### Documentation
- ✅ `docs/EXPERIMENT_6_REFACTORING.md` - Comprehensive refactoring documentation

### Testing & Examples
- ✅ `tests/test_experiment_6.py` - Validation test suite
- ✅ `examples/experiment_6_usage.py` - Interactive usage examples

### Notebooks
- ✅ `notebooks/exer_recog/06_pose_mlp.ipynb` - Updated with config-based approach documentation

---

## 🚀 Quick Start

### 1. Single Run (Config-Based)
```python
from src.scripts.experiment_6 import train_experiment_6_static

results = train_experiment_6_static(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)
```

### 2. Multi-Run (30 runs)
```python
from src.scripts.experiment_6 import train_experiment_6_multi_run

all_runs, stats = train_experiment_6_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)

print(f"Accuracy: {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
```

### 3. Legacy Style (Backward Compatible)
```python
results = train_experiment_6_static(
    npz_path='datasets/Mediapipe pose estimates/pose_data_side_static.npz',
    config_path='config/experiment_6.yaml',
    seed=99,
    batch_size=32,
    max_epochs=100
)
```

---

## 📊 Configuration Overview

### Key Config Sections

**Dataset Splits:**
```yaml
dataset:
  val_ratio: 0.15      # 15% subjects for validation
  test_ratio: 0.30     # 30% subjects for test
  random_seed: 42      # Base random seed
  stratified: true     # Stratified subject splits
```

**Model Architecture:**
```yaml
model:
  hidden_sizes: [256, 128]  # MLP layer widths
  dropout: 0.25             # Dropout rate
  num_classes: 15           # Number of exercise classes
```

**Training Hyperparameters:**
```yaml
training:
  batch_size: 64      # Batch size
  lr: 0.001          # Learning rate
  max_epochs: 80     # Maximum epochs
  optimizer: adam    # Optimizer type
```

**Multi-Run Settings:**
```yaml
multi_run:
  enabled: true      # Enable/disable multi-run
  num_runs: 30       # Number of runs
  base_seed: 42      # Base seed (run i uses base_seed + i)
```

---

## 🧪 Testing

### Run Validation Tests
```bash
cd /mnt/d/Graduation_Project/ai-virtual-coach
python tests/test_experiment_6.py
```

### Run Interactive Examples
```bash
python examples/experiment_6_usage.py
```

---

## 📂 Results Directory Structure

### Single Run
```
experiments/exer_recog/results/exp_06_pose_mlp_static/
└── run_001/
    └── metrics.json
```

### Multi-Run
```
experiments/exer_recog/results/
└── multi_run_001/
    ├── config.yaml              # Config snapshot
    ├── aggregated_stats.json    # Statistics (JSON)
    ├── aggregated_summary.txt   # Statistics (human-readable)
    ├── all_runs.json           # All run results
    ├── run_001/
    │   └── metrics.json
    ├── run_002/
    │   └── metrics.json
    └── ...
```

---

## 🔧 Common Customizations

### Change Number of Runs
Edit `config/experiment_6.yaml`:
```yaml
multi_run:
  num_runs: 10  # Faster experiments
```

### Adjust Model Depth
```yaml
model:
  hidden_sizes: [512, 256, 128, 64]  # 4-layer network
  dropout: 0.35                       # Higher regularization
```

### Modify Results Location
```yaml
results:
  base_dir: /path/to/custom/results
```

### Adjust Early Stopping
```yaml
callbacks:
  early_stopping:
    patience: 15      # Wait longer
    min_delta: 0.001  # Stricter improvement threshold
```

---

## 🎯 Typical Workflows

### 1. Quick Experiment (Single Run)
```python
# Test a hypothesis quickly
results = train_experiment_6_static(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml',
    max_epochs=20  # Quick training
)
```

### 2. Statistical Validation (Multi-Run)
```python
# Get robust performance estimates
all_runs, stats = train_experiment_6_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)

# Report with confidence intervals
print(f"Test Macro F1: {stats['test_macro_f1']['mean']:.4f} ± {stats['test_macro_f1']['std']:.4f}")
```

### 3. Compare Views (Front vs Side)
```python
# Train on front view
front_runs, front_stats = train_experiment_6_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_static.npz',
    config_path='config/experiment_6.yaml'
)

# Train on side view  
side_runs, side_stats = train_experiment_6_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_side_static.npz',
    config_path='config/experiment_6.yaml'
)

# Compare
print(f"Front: {front_stats['test_macro_f1']['mean']:.4f} ± {front_stats['test_macro_f1']['std']:.4f}")
print(f"Side:  {side_stats['test_macro_f1']['mean']:.4f} ± {side_stats['test_macro_f1']['std']:.4f}")
```

---

## 🔍 Accessing Results

### From Python
```python
# After single run
print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Per-class F1: {results['test_metrics']['per_class_f1']}")

# After multi-run
print(f"Mean Accuracy: {stats['test_accuracy']['mean']:.4f}")
print(f"Std Accuracy: {stats['test_accuracy']['std']:.4f}")
```

### From Saved Files
```python
import json

# Load single run metrics
with open('experiments/.../run_001/metrics.json', 'r') as f:
    metrics = json.load(f)

# Load multi-run aggregated stats
with open('experiments/.../multi_run_001/aggregated_stats.json', 'r') as f:
    stats = json.load(f)
```

---

## 🎓 Key Features

✅ **YAML Configuration:** All hyperparameters in version-controlled config files  
✅ **Multi-Run Support:** Statistical validation with automatic aggregation  
✅ **Backward Compatible:** Existing code continues to work  
✅ **Auto-Incrementing Folders:** Never overwrite previous results  
✅ **Memory Management:** Automatic cleanup between runs  
✅ **Comprehensive Logging:** Detailed progress and result tracking  
✅ **Flexible:** Override config with function parameters  
✅ **Future-Ready:** Temporal config prepared for future features

---

## 📞 Need Help?

- **Full Documentation:** See `docs/EXPERIMENT_6_REFACTORING.md`
- **Test Suite:** Run `tests/test_experiment_6.py`
- **Interactive Examples:** Run `examples/experiment_6_usage.py`
- **Notebook Demo:** Open `notebooks/exer_recog/06_pose_mlp.ipynb`

---

**Last Updated:** December 21, 2025
