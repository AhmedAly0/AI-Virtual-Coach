# Experiment 1 Quick Reference Guide (Pose MLP)

## 📁 Files

### Configuration Files
- ✅ `config/experiment_1_baseline_front.yaml` - Baseline 19 features (front view)
- ✅ `config/experiment_1_baseline_side.yaml` - Baseline 19 features (side view)
- ✅ `config/experiment_1_specialized_front.yaml` - Specialized 37 features (front view)
- ✅ `config/experiment_1_specialized_side.yaml` - Specialized 37 features (side view)

### Source Code
- ✅ `src/scripts/experiment_1.py` - Pose MLP with YAML config and multi-run support

### Testing
- ✅ `tests/test_experiment_1.py` - Validation test suite

### Notebooks
- ✅ `notebooks/exer_recog/01_pose_mlp.ipynb` - Training, analysis, and comparison

---

## 🚀 Quick Start

### 1. Single Run (Config-Based)
```python
from src.scripts.experiment_1 import train_experiment_1

results = train_experiment_1(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml'
)
```

### 2. Multi-Run (30 runs)
```python
from src.scripts.experiment_1 import train_experiment_1_multi_run

all_runs, stats = train_experiment_1_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml'
)

print(f"Accuracy: {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
```

### 3. Legacy Style (Backward Compatible)
```python
results = train_experiment_1(
    npz_path='datasets/Mediapipe pose estimates/pose_data_side_19_features.npz',
    config_path='config/experiment_1_baseline_side.yaml',
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
  feature_type: 'all'  # 'angles', 'distances', 'all', 'base_specialized'
```

**Model Architecture:**
```yaml
model:
  hidden_sizes: [512, 256, 128]  # MLP layer widths
  dropout: 0.35                  # Dropout rate
  num_classes: 15                # Number of exercise classes
```

**Training Hyperparameters:**
```yaml
training:
  batch_size: 16       # Batch size
  lr: 0.00006         # Learning rate
  max_epochs: 200     # Maximum epochs
  optimizer: adam      # Optimizer type
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
cd "/mnt/d/Graduation Project/ai-virtual-coach"
python tests/test_experiment_1.py
```

---

## 📂 Results Directory Structure

### Multi-Run
```
output/exer_recog/
├── exp_01_pose_mlp_baseline/
│   ├── front/
│   │   └── multi_run_001/
│   │       ├── config.yaml
│   │       ├── aggregated_stats.json
│   │       ├── aggregated_summary.txt
│   │       ├── all_runs.json
│   │       ├── run_001/
│   │       │   ├── metrics.json
│   │       │   └── model.keras
│   │       └── ...
│   └── side/
│       └── multi_run_001/
└── exp_01_pose_mlp_specialized/
    ├── front/
    └── side/
```

---

## 🎯 Typical Workflows

### 1. Quick Experiment (Single Run)
```python
results = train_experiment_1(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml',
    max_epochs=20  # Quick training
)
```

### 2. Statistical Validation (Multi-Run)
```python
all_runs, stats = train_experiment_1_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml'
)

print(f"Test Macro F1: {stats['test_macro_f1']['mean']:.4f} ± {stats['test_macro_f1']['std']:.4f}")
```

### 3. Specialized Features Experiment
```python
all_runs, stats = train_experiment_1_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_specialized_front.yaml'
)
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
print(f"Best Run F1: {stats['best_run']['test_macro_f1']:.4f}")
```

### From Saved Files
```python
import json
with open('output/exer_recog/exp_01_pose_mlp_baseline/front/multi_run_001/aggregated_stats.json') as f:
    stats = json.load(f)
```

---

## 🎓 Key Features

✅ **YAML Configuration:** All hyperparameters in version-controlled config files  
✅ **Multi-Run Support:** Statistical validation with automatic aggregation  
✅ **Backward Compatible:** Legacy `train_experiment_6` aliases still work  
✅ **Auto-Incrementing Folders:** Never overwrite previous results  
✅ **Memory Management:** Automatic cleanup between runs  
✅ **Feature Selection:** Baseline (19) or specialized (37) feature sets  
✅ **Dual View:** Front and side camera configs  

---

## 📝 Migration from Experiment 6

If you have code referencing the old experiment 6 names:

| Old Name | New Name |
|----------|----------|
| `experiment_6.py` | `experiment_1.py` |
| `train_experiment_6()` | `train_experiment_1()` |
| `train_experiment_6_multi_run()` | `train_experiment_1_multi_run()` |
| `experiment_6_temporal_front.yaml` | `experiment_1_baseline_front.yaml` |
| `experiment_6_ablation_specialized_front.yaml` | `experiment_1_specialized_front.yaml` |
| `exp_06_pose_mlp_temporal/` | `exp_01_pose_mlp_baseline/` |
| `exp_06_ablation_specialized/` | `exp_01_pose_mlp_specialized/` |

> **Note:** Backward-compatible aliases (`train_experiment_6`, `train_experiment_6_multi_run`) are defined in `experiment_1.py` and will continue to work but are deprecated.

---

**Last Updated:** February 8, 2026
