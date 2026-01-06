# GEI Exercise Recognition - Refactored Structure

## 🎯 Project Overview
This project implements deep learning-based exercise recognition using Gait Energy Images (GEI) with two distinct experimental approaches comparing different transfer learning strategies.

## 📁 Project Structure

```
ai-virtual-coach/
├── config/                          # Experiment configurations
│   ├── experiment_1.yaml           # 2-phase training config
│   └── experiment_2.yaml           # 3-stage progressive config
│
├── datasets/                        # Data directory
│   ├── GEIs_of_rgb_front/
│   │   └── GEIs/                   # Main GEI dataset
│   ├── Filtered clips/             # Video clips
│   └── metadata.xlsx.csv
│
├── experiments/                     # Experiment results
│   └── results/
│       ├── exp_01_baseline/        # Experiment 1 results
│       │   ├── README.md
│       │   ├── efficientnet_b0/
│       │   ├── resnet50/
│       │   └── ...
│       └── exp_02_progressive/     # Experiment 2 results
│           ├── README.md
│           ├── efficientnet_b0/
│           ├── resnet50/
│           └── ...
│
├── notebooks/                       # Jupyter notebooks
│   ├── exer_reccog_gei_tf.ipynb   # Main experiment notebook
│   └── EDA/
│       └── Analysis.ipynb
│
├── src/                            # Source code (MODULAR STRUCTURE)
│   ├── data/                       # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py         # load_data, split_by_subject
│   │   ├── preprocessing.py       # prep_tensors, resize, normalize
│   │   ├── augmentation.py        # data_augmentater, configs
│   │   └── dataset_builder.py     # build_datasets, make_split
│   │
│   ├── models/                     # Model architectures
│   │   ├── __init__.py
│   │   └── model_builder.py       # build_model_*, BACKBONE_REGISTRY
│   │
│   ├── training/                   # Training experiments
│   │   ├── __init__.py
│   │   ├── experiment_1.py        # 2-phase training
│   │   └── experiment_2.py        # 3-stage progressive
│   │
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── io_utils.py            # Folder setup, seed setting
│   │   ├── metrics.py             # Experiment tracking
│   │   └── visualization.py       # Plots and comparisons
│   │
│   └── Training/                   # Legacy code (DEPRECATED)
│       ├── gei_lib_tf.py          # ⚠️ DEPRECATED - Use experiment_2.py
│       ├── gei_lib_tf_v2.py       # ⚠️ DEPRECATED - Use experiment_1.py
│       └── deprecated/
│
└── results/                        # Old results (to be migrated)
    ├── efficientnet_b0/
    ├── resnet50/
    └── ...
```

## 🔬 Experiments

### Experiment 1: 2-Phase Transfer Learning (Baseline)
**Location**: `src/scripts/experiment_1.py`  
**Config**: `config/experiment_1.yaml`  
**Results**: `experiments/results/exp_01_baseline/`

**Strategy**:
- Phase 1: Frozen backbone (10 epochs)
- Phase 2: Full unfreeze (50 epochs)
- ✅ Validation monitoring (EarlyStopping, ReduceLROnPlateau)
- ✅ Basic augmentation (flip, translation)
- ✅ Standard classification head

### Experiment 2: 3-Stage Progressive Unfreezing (Advanced)
**Location**: `src/scripts/experiment_2.py`  
**Config**: `config/experiment_2.yaml`  
**Results**: `experiments/results/exp_02_progressive/`

**Strategy**:
- Stage 1: Frozen (15-35 epochs)
- Stage 2: 10% unfrozen (8-20 epochs)
- Stage 3: 30% unfrozen (17-25 epochs)
- ✅ Blind training (no validation monitoring)
- ✅ Enhanced augmentation (flip, translation, rotation, zoom, brightness)
- ✅ Architecture-specific heads
- ✅ Per-backbone epoch tuning

## 🚀 Quick Start

### Basic Usage

```python
from src.data import load_data
from src.training import train_experiment_1, train_experiment_2
from src.utils import set_global_seed

# Set reproducibility
set_global_seed(42)

# Load dataset
dataset = load_data('datasets/GEIs_of_rgb_front/GEIs')

# Define backbones
backbones = [
    'efficientnet_b0',
    'resnet50',
    'mobilenet_v2',
    'vgg16'
]

# Run Experiment 1 (Baseline)
exp1_results = train_experiment_1(
    dataset=dataset,
    backbones=backbones,
    num_runs=3,
    test_ratio=0.3
)

# Run Experiment 2 (Progressive)
exp2_results = train_experiment_2(
    dataset=dataset,
    backbones=backbones,
    num_runs=3,
    test_ratio=0.3
)
```

### Comparing Results

```python
from src.utils import (
    load_backbone_results_with_config,
    get_all_model_parameters,
    create_comprehensive_comparison,
    generate_statistical_comparison
)

# Load results
exp1_results = load_backbone_results_with_config('experiments/results/exp_01_baseline')
exp2_results = load_backbone_results_with_config('experiments/results/exp_02_progressive')

# Get model parameters
params_df = get_all_model_parameters(backbones)

# Create comparison
create_comprehensive_comparison(exp1_results, params_df, 'experiments/comparisons/exp1')
generate_statistical_comparison(exp1_results, 'experiments/comparisons/exp1')
```

## 📊 Supported Backbones

| Backbone | Parameters | Input Size |
|----------|-----------|------------|
| EfficientNetV2 B0 | 7.1M | 224×224 |
| EfficientNetV2 B2 | 10.1M | 224×224 |
| EfficientNetV2 B3 | 14.4M | 224×224 |
| ResNet50 | 25.6M | 224×224 |
| VGG16 | 138.4M | 224×224 |
| MobileNetV2 | 3.5M | 224×224 |
| MobileNetV3 Large | 5.4M | 224×224 |

## 🔧 Module Documentation

### src/data/
- **data_loader.py**: Load GEI images from nested folder structure
- **preprocessing.py**: Image preprocessing (resize, normalize, tensor conversion)
- **augmentation.py**: Data augmentation pipelines (basic & enhanced)
- **dataset_builder.py**: Build tf.data.Dataset with augmentation

### src/models/
- **model_builder.py**: Build all model architectures
  - `build_model()`: Simple CNN
  - `build_model_for_backbone()`: Standard transfer learning (Experiment 1)
  - `build_model_for_backbone_v2()`: Architecture-specific heads (Experiment 2)
  - `get_callbacks()`: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### src/scripts/
- **experiment_1.py**: 2-phase training with validation monitoring
- **experiment_2.py**: 3-stage progressive blind training

### src/utils/
- **io_utils.py**: Folder management, seed setting
- **metrics.py**: Experiment tracking, parameter counting
- **visualization.py**: Training curves, confusion matrices, comparisons

## ⚠️ Migration from Legacy Code

### Old Code (DEPRECATED)
```python
import gei_lib_tf as lib  # ❌ DEPRECATED

dataset = lib.load_data('datasets/GEIs_of_rgb_front/GEIs')
results = lib.train_one_run_progressive(...)
```

### New Code (RECOMMENDED)
```python
from src.data import load_data  # ✅ MODULAR
from src.training import train_one_run_progressive

dataset = load_data('datasets/GEIs_of_rgb_front/GEIs')
results = train_one_run_progressive(...)
```

## 📝 Configuration Files

### experiment_1.yaml
```yaml
training:
  frozen_epochs: 10
  fine_tune_epochs: 50
  batch_size: 32

augmentation:
  horizontal_flip: true
  translation: 0.15
  rotation: false  # Experiment 1 specific
```

### experiment_2.yaml
```yaml
training:
  progressive_unfreezing: true
  unfreeze_stage_1_percent: 0.10
  unfreeze_stage_2_percent: 0.30

epochs:  # Per-backbone tuning
  efficientnet_b0: {frozen: 25, stage1: 15, stage2: 25}
  resnet50: {frozen: 20, stage1: 10, stage2: 20}

augmentation:
  rotation: true  # ±18 degrees
  zoom: true      # ±10%
  brightness: true  # ±10%
```

## 🔍 Key Differences Between Experiments

| Feature | Experiment 1 | Experiment 2 |
|---------|-------------|--------------|
| **Phases** | 2 | 3 |
| **Unfreezing** | 0% → 100% | 0% → 10% → 30% |
| **Validation** | Monitored | Blind |
| **Augmentation** | Basic (2) | Enhanced (5) |
| **Epochs** | Fixed (60) | Per-backbone (40-80) |
| **Heads** | Standard | Architecture-specific |
| **Callbacks** | EarlyStopping + ReduceLR | ReduceLR only |

## 📦 Dependencies

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.66.0
pyyaml>=6.0.0
```

## 🧪 Testing

```python
# Test data loading
from src.data import load_data
dataset = load_data('datasets/GEIs_of_rgb_front/GEIs')
print(f"Loaded {len(dataset)} samples")

# Test model building
from src.models import build_model_for_backbone
model, preprocess_fn = build_model_for_backbone('resnet50', 224, 15)
print(f"Model: {model.count_params()} parameters")

# Test preprocessing
from src.data import prep_tensors
X_train, y_train = prep_tensors(dataset[:10], {}, 224)
print(f"Tensors: {X_train.shape}, {y_train.shape}")
```

## 📄 License
[Your License Here]

## 👤 Author
Ahmed Mohamed Ahmed
October 2025

## 🙏 Acknowledgments
- TensorFlow/Keras team for pretrained models
- Research paper authors for GEI methodology
