# Experiment 4 Implementation Summary

## Created Files

1. **`src/models/model_builder.py`** - Added `build_model_for_backbone_v4()` function
2. **`src/scripts/experiment_4.py`** - Training script with AdamW optimizer
3. **`config/experiment_4.yaml`** - Configuration file (same structure as experiment_3.yaml)
4. **`notebooks/exer_recog/04_regularized.ipynb`** - Jupyter notebook for running experiments

## Key Features

### 1. Dual Pooling
- Concatenates GlobalAveragePooling2D + GlobalMaxPooling2D
- Captures both average patterns (GAP) and salient features (GMP)
- Doubles feature dimension after backbone

### 2. Smaller Classification Heads
**EfficientNet (B0/B2/B3):**
- Dual pooling → BN → Dense(256, swish) + Drop(0.3) → Dense(128, swish) + Drop(0.2)
- Reduced from 512→256 (Exp 3) to 256→128

**ResNet50 / VGG16:**
- Dual pooling → BN → Dense(256, relu) + Drop(0.4)
- Simplified from two-layer heads in Exp 3

**MobileNet (V2/V3-Large):**
- Dual pooling → Dense(128, relu) + Drop(0.25)
- Single layer (vs. 256→128 in Exp 3)

### 3. Label Smoothing
- Factor: 0.1
- Correct class: 0.9, Other classes: 0.1/14 each
- Prevents overconfident predictions

### 4. AdamW Optimizer
- Weight decay: 1e-4
- Better decoupling of weight decay and learning rate
- More stable than L2 regularization

### 5. Differential Learning Rates
- Head: 1e-4 (higher for new layers)
- Backbone: 1e-5 (lower to preserve pretrained features)
- Prevents catastrophic forgetting

## Configuration (experiment_4.yaml)

Now follows the same structure as experiment_3.yaml:

```yaml
training:
  strategy: "2-phase-validation-monitored-regularized-dual-pooling"
  frozen_epochs: 30
  fine_tune_epochs: 20
  batch_size: 32
  initial_lr: 0.0001      # Head learning rate (AdamW)
  fine_tune_lr: 0.00001   # Backbone learning rate (AdamW)
  weight_decay: 0.0001    # AdamW weight decay

model:
  img_size: 224
  num_classes: 15
  classification_head: "architecture-specific-v4-dual-pooling-regularized"
  label_smoothing: 0.1

unfreezing:
  strategy: "uniform-percentage"  # 50% for all backbones
  efficientnet_b0:
    phase2_unfreeze_percent: 0.50
  # ... (all backbones use 0.50)
```

## Usage

### Option 1: Run via notebook
```python
# Open notebooks/exer_recog/04_regularized.ipynb
# Execute cells sequentially
```

### Option 2: Run via Python script
```python
from src.scripts.experiment_4 import train_experiment_4

# Run all backbones
train_experiment_4(num_runs=5)

# Run specific backbones
train_experiment_4(
    backbones=['efficientnet_b0', 'resnet50'],
    num_runs=3
)
```

### Option 3: Command line
```bash
cd "d:\Graduation Project\ai-virtual-coach"
python -m src.scripts.experiment_4
```

## Results Structure

```
experiments/exer_recog/results/
└── exp_04_regularized/
    ├── efficientnet_b0/
    │   ├── run_001/
    │   │   ├── best_model.keras
    │   │   └── results.yaml
    │   ├── run_002/
    │   └── run_003/
    ├── efficientnet_b2/
    ├── resnet50/
    └── ...
```

## Expected Improvements over Experiment 3

1. **Better generalization**: Label smoothing + smaller heads reduce overfitting
2. **Richer features**: Dual pooling captures more diverse motion patterns
3. **Lower variance**: More consistent results across runs
4. **Stable training**: AdamW prevents catastrophic forgetting

## Comparison with Previous Experiments

| Feature | Exp 1 | Exp 3 | Exp 4 |
|---------|-------|-------|-------|
| Pooling | GAP | GAP | GAP + GMP |
| Head Design | Universal | Architecture-specific | Architecture-specific (smaller) |
| Head Layers | 1 | 2 | 1-2 (smaller) |
| Optimizer | Adam | Adam | AdamW |
| Label Smoothing | No | No | Yes (0.1) |
| Differential LR | No | No | Yes |
| Weight Decay | No | No | Yes (1e-4) |

## Next Steps

1. Run the notebook to train models
2. Compare results with Experiment 3
3. Analyze variance reduction
4. Evaluate on test set
5. Generate comparison visualizations

## Notes

- All files follow the same structure as Experiments 1-3
- Compatible with existing utility functions (io_utils, metrics, etc.)
- Results folders use new naming: `run_001`, `run_002`, etc. (no timestamps)
- Progressive unfreezing (50%) same as Experiment 3
