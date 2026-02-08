# ⚠️ DEPRECATED — See [README.md](../README.md) instead

This document has been **consolidated into the unified [README.md](../README.md)** at the repository root.

## Content Moved

- **Data Splitting Methodology** → [README.md § Subject-Wise Data Splitting (Deep Dive)](../README.md#subject-wise-data-splitting)
- **Rationale & Validation** → [README.md § Subject-Wise Data Splitting](../README.md#subject-wise-data-splitting)
- **Implementation Details** → [README.md § Subject-Wise Data Splitting](../README.md#subject-wise-data-splitting)

## Why Consolidated?

The four separate documentation files have been merged into a single comprehensive README that serves as both a GitHub landing page and a complete technical reference.

**Please refer to [README.md](../README.md) for all information.**

---

_This file is deprecated and no longer maintained. [Click here to view the main README](../README.md)._

This document describes the **subject-wise splitting approach** used across all exercise recognition experiments to ensure proper generalization testing and prevent data leakage. This methodology is fundamental to evaluating whether our models learn exercise movements rather than individual body characteristics.

---

## Why Subject-Wise Splitting?

### The Problem: Data Leakage from Sample-Level Splitting

Traditional **sample-level splitting** randomly divides all GEI images into train/validation/test sets, ignoring which subject performed each exercise. This creates a critical problem:

- **Same subject appears in multiple splits**: Subject A's GEIs are in both training and test sets
- **Model learns body shapes, not exercises**: The model recognizes "this is Subject A doing any exercise" instead of "this is a shoulder press movement"
- **Inflated performance metrics**: Test accuracy appears high because the model already saw this person's body shape during training
- **Fails in deployment**: When encountering completely new users, performance drops dramatically

### The Solution: Subject-Wise Splitting

**Subject-wise splitting** ensures that all samples (GEIs) from the same subject are assigned to **exactly ONE split** (train, validation, OR test) - never distributed across multiple splits.

**Key Principle**: 
```
IF subject_id = "volunteer_007" is in training set
THEN ALL GEIs of volunteer_007 are in training set
     AND volunteer_007 has ZERO GEIs in validation or test sets
```

**Benefits**:
1. **Simulates real-world deployment**: Test subjects are completely unseen during training
2. **Tests true generalization**: Model must learn exercise movements, not body shapes
3. **Prevents data leakage**: No information about test subjects leaks into training
4. **Realistic performance metrics**: Test accuracy reflects performance on new users

---

## Core Splitting Functions

All splitting logic is implemented in [src/data/data_loader.py](src/data/data_loader.py) and wrapped by convenience functions in [src/data/dataset_builder.py](src/data/dataset_builder.py).

### 1. Two-Way Split: `split_by_subject_two_way()`

**Purpose**: Split dataset into **train** and **test** sets by assigning subjects to one split or the other.

**Signature**:
```python
def split_by_subject_two_way(
    dataset: List[Tuple[str, np.ndarray, str]],
    split_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List, List]
```

**Parameters**:
- `dataset`: List of `(exercise_name, image, subject_id)` tuples
- `split_ratio`: Fraction of subjects to assign to the smaller split (default 0.3 = 30% test)
- `seed`: Random seed for reproducible splits
- `stratified`: If True, ensures all exercise classes appear in both splits

**Returns**: `(train_samples, test_samples)`

**Example**:
```python
from src.data import split_by_subject_two_way

# Random split (stratified=False)
train, test = split_by_subject_two_way(
    dataset, 
    split_ratio=0.3, 
    seed=42, 
    stratified=False
)
# Result: ~70% subjects in train, ~30% in test
# Some exercises may be missing from test set

# Stratified split (stratified=True)
train, test = split_by_subject_two_way(
    dataset, 
    split_ratio=0.3, 
    seed=42, 
    stratified=True
)
# Result: ~70% subjects in train, ~30% in test
# ALL 15 exercises guaranteed in both train and test
```

---

### 2. Three-Way Split: `split_by_subjects_three_way()`

**Purpose**: Split dataset into **train**, **validation**, and **test** sets for experiments that need a dedicated validation set for early stopping.

**Signature**:
```python
def split_by_subjects_three_way(
    dataset: List[Tuple[str, np.ndarray, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List, List, List]
```

**Parameters**:
- `dataset`: List of `(exercise_name, image, subject_id)` tuples
- `val_ratio`: Fraction of subjects for validation (default 0.15 = 15%)
- `test_ratio`: Fraction of subjects for test (default 0.3 = 30%)
- `seed`: Random seed for reproducibility
- `stratified`: If True, ensures all classes in all three splits

**Returns**: `(train_samples, val_samples, test_samples)`

**Example**:
```python
from src.data import split_by_subjects_three_way

# Stratified 3-way split
train, val, test = split_by_subjects_three_way(
    dataset,
    val_ratio=0.15,   # 15% subjects for validation
    test_ratio=0.3,   # 30% subjects for test
    seed=42,
    stratified=True   # All 15 classes in all 3 splits
)
# Result: ~55% train, ~15% val, ~30% test
```

**When to Use**:
- ✅ Single-run experiments with early stopping (no cross-validation)
- ✅ Multi-run experiments with different seeds (each run creates new splits)
- ❌ K-fold cross-validation (use two-way split + fold builder instead)

---

### 3. K-Fold Subject-Wise Folds: `build_subject_folds()`

**Purpose**: Create K subject-disjoint folds for **K-fold cross-validation**, where each fold contains completely different subjects.

**Signature**:
```python
def build_subject_folds(
    dataset: List[Tuple[str, np.ndarray, str]],
    num_folds: int = 5,
    seed: int = None,
    stratified: bool = True
) -> List[List[Tuple]]
```

**Parameters**:
- `dataset`: List of `(exercise_name, image, subject_id)` tuples
- `num_folds`: Number of folds to create (default 5)
- `seed`: Random seed for reproducible fold assignment
- `stratified`: If True, balances both subject count and class distribution

**Returns**: List of K folds, each fold is a list of samples

**Example**:
```python
from src.data import build_subject_folds

# Create 5 subject-wise folds
folds = build_subject_folds(
    dataset,
    num_folds=5,
    seed=42,
    stratified=True
)

# Use for K-fold CV
for fold_idx, val_fold in enumerate(folds):
    train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
    train_samples = sum(train_folds, [])  # Concatenate all training folds
    val_samples = val_fold
    
    # Train model on train_samples, evaluate on val_samples
```

**Fold Properties**:
- **Subject-disjoint**: Each fold contains completely different subjects
- **Balanced size**: Folds have approximately equal number of samples
- **Balanced classes** (when `stratified=True`): Folds have similar label distributions
- **Deterministic**: Same seed always produces same fold assignments

---

## Stratification Algorithm

Stratification ensures that **all exercise classes appear in all splits** while maintaining subject integrity. This is critical because some exercises have very few subjects who performed them.

### Why Stratification?

**Problem without stratification**:
```
Dataset: 49 subjects, 15 exercises
Exercise "Deadlift": Only 2 subjects performed it

Random 70/30 split:
- Train: 34 subjects (may include both Deadlift subjects)
- Test:  15 subjects (may include ZERO Deadlift subjects)

Result: Model trained on Deadlift but never tested on it
        → Cannot measure model's ability to recognize Deadlifts
```

**Solution with stratification**:
```
Exercise "Deadlift": 2 subjects

Stratified 70/30 split:
- Train: 1 Deadlift subject  (50% of Deadlift subjects)
- Test:  1 Deadlift subject  (50% of Deadlift subjects)

Result: Model trained on 1 Deadlift subject, tested on different Deadlift subject
        → Can measure generalization on Deadlifts
```

### Stratification Algorithm (Two-Way Split)

```
INPUT: 
  - dataset: List of (exercise, image, subject_id) tuples
  - split_ratio: Fraction for smaller split (e.g., 0.3 for 30% test)
  - seed: Random seed

ALGORITHM:

1. Build subject→exercises mapping
   - Maps each subject to list of exercises they performed
   - Handles subjects who performed multiple exercises
   
2. Build exercise→subjects reverse mapping
   - Maps each exercise to list of subjects who performed it
   - Deduplicate: same subject counted once per exercise

3. Initialize tracking sets:
   - set1_subjects = {}  # Larger split (e.g., train)
   - set2_subjects = {}  # Smaller split (e.g., test)
   - assigned_subjects = {}  # Global tracker to prevent duplication

4. FOR EACH exercise class:
   a. Get subjects who performed this exercise
   b. Filter out already-assigned subjects
   c. Calculate target: n_set2 = max(1, available * split_ratio)
   d. Assign subjects:
      - set2 ← available[:n_set2]  # Smaller split
      - set1 ← available[n_set2:]  # Larger split
   e. Update assigned_subjects to include both sets
   
5. Handle remaining unassigned subjects:
   - Assign to larger split (set1)

6. VERIFY: No overlap between set1_subjects and set2_subjects
   - Raises RuntimeError if overlap detected

7. Partition dataset by subject membership:
   - samples_set1 = all samples where subject_id in set1_subjects
   - samples_set2 = all samples where subject_id in set2_subjects

OUTPUT: (samples_set1, samples_set2)
```

### Edge Case Handling

| Scenario | Behavior | Example |
|----------|----------|---------|
| **Exercise with 1 subject** | Assigned to larger split (logged as warning) | "Deadlift" has 1 subject → goes to train only |
| **Exercise with 2 subjects** | 1 subject per split | "Calf raises" has 2 subjects → 1 train, 1 test |
| **Exercise with 10+ subjects** | Proportional split (e.g., 7 train, 3 test) | "Shoulder press" has 10 subjects → 7 train, 3 test |
| **Subject performs multiple exercises** | ALL exercises of that subject go to same split | Volunteer_007 did 5 exercises → all 5 in same split |

### Three-Way Stratification (Train/Val/Test)

The three-way stratification extends the two-way algorithm to three splits with an additional constraint: **ALL classes must appear in validation split** (not just train and test).

**Priority Order**:
1. **Test set** (30%): First pass assigns subjects to ensure all classes present
2. **Validation set** (15%): Second pass assigns from remaining subjects
3. **Train set** (~55%): Gets all remaining subjects

**Guarantees**:
- ✅ All 15 exercise classes in test set (critical for final evaluation)
- ✅ All 15 exercise classes in train set (required for learning)
- ⚠️ Best-effort for validation set (may miss 1-2 classes if too few subjects)

---

## Subject ID Normalization

To handle inconsistent folder naming across the dataset, all subject identifiers are **normalized** before splitting.

### Why Normalize?

**Problem**: Dataset folders have inconsistent naming:
```
- "Volunteer #1"
- "volunteer #10"
- "v20"
- "Volunteer_31"
```

**Solution**: Canonicalize all subject IDs to a standard format.

### Normalization Rules

```python
def _normalize_subject_id(subject_id: str) -> str:
    """Convert any folder name to 'volunteer_XXX' format."""
    
    # Extract numeric part if present
    match = re.search(r"(\d+)", subject_id)
    if match:
        number = match.group(1).zfill(3)  # Zero-pad to 3 digits
        return f"volunteer_{number}"
    
    # If no number, slugify the string
    slug = re.sub(r"[^a-z0-9]+", "_", subject_id.lower())
    return slug or "volunteer_unknown"
```

**Examples**:
```
"Volunteer #1"     → "volunteer_001"
"volunteer #10"    → "volunteer_010"
"v20"              → "volunteer_020"
"Volunteer_31"     → "volunteer_031"
"John Doe"         → "john_doe"
""                 → "volunteer_unknown"
```

**Impact**: All splitting functions use normalized IDs internally, ensuring consistent subject assignment across experiments.

---

## Verification: Detecting Data Leakage

The `verify_subject_split_integrity()` utility function checks for data leakage after splitting.

### Usage

```python
from src.data import verify_subject_split_integrity

# After splitting
train, val, test = split_by_subjects_three_way(dataset, ...)

# Verify no leakage
results = verify_subject_split_integrity(
    train, val, test, verbose=True
)

if results['has_subject_overlap']:
    raise RuntimeError("Data leakage detected!")
```

### What It Checks

1. **Subject Overlap**: 
   - `train ∩ val = ∅` (no shared subjects)
   - `train ∩ test = ∅`
   - `val ∩ test = ∅`

2. **Subject Count**:
   - Total unique subjects = sum of subjects in all splits
   - Verifies no subjects were lost or duplicated

3. **Class Coverage**:
   - Which exercises appear in each split
   - Warns if any split is missing classes

4. **Sample Distribution**:
   - Number of samples per split
   - Checks for reasonable proportions

### Output

```python
{
    'has_subject_overlap': False,  # ✅ No leakage
    'total_unique_subjects': 49,
    'train_subjects_count': 27,
    'val_subjects_count': 7,
    'test_subjects_count': 15,
    'train_classes': 15,
    'val_classes': 15,
    'test_classes': 15,
    'total_classes': 15,
    # ... detailed per-split statistics
}
```

---

## Usage Patterns by Experiment Type

### Pattern 1: Simple Train/Test Split (Experiments 1-2)

**Use Case**: Basic supervised learning with single train/test split.

```python
from src.data.dataset_builder import make_split

# Wrapper around split_by_subject_two_way()
train_samples, test_samples, label_to_int = make_split(
    dataset,
    test_ratio=0.3,
    seed=42,
    stratified=True  # All classes in both splits
)
```

**Example Experiments**: 
- Experiment 1: Transfer learning with ImageNet backbones
- Experiment 2: Progressive training strategies

---

### Pattern 2: Three-Way Split for Early Stopping (Experiments 3-4)

**Use Case**: Need dedicated validation set for hyperparameter tuning and early stopping.

```python
from src.data.dataset_builder import make_split_three_way

train_samples, val_samples, test_samples, label_to_int = make_split_three_way(
    dataset,
    val_ratio=0.15,
    test_ratio=0.3,
    seed=42,
    stratified=True  # All classes in all three splits
)

# Train with early stopping on val_samples
# Final evaluation on test_samples (never seen during training)
```

**Example Experiments**:
- Experiment 3: Smart multi-head architectures
- Experiment 4: Regularized models with dropout/weight decay

---

### Pattern 3: K-Fold Cross-Validation (Experiment 5)

**Use Case**: Robust hyperparameter validation using multiple train/val splits.

```python
from src.data import split_by_subject_two_way, build_subject_folds

# Step 1: Create frozen test set (30%)
pool_samples, test_samples = split_by_subject_two_way(
    dataset,
    split_ratio=0.3,
    seed=42,
    stratified=True
)

# Step 2: Create K folds from remaining 70%
folds = build_subject_folds(
    pool_samples,
    num_folds=5,
    seed=42,
    stratified=True
)

# Step 3: K-fold cross-validation
for fold_idx, val_fold in enumerate(folds):
    train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
    train_samples = sum(train_folds, [])
    val_samples = val_fold
    
    # Train and validate
    model = train_model(train_samples, val_samples)
    
# Step 4: Retrain on full pool, evaluate on frozen test
final_model = retrain_model(pool_samples)
test_metrics = evaluate(final_model, test_samples)
```

**Advantages**:
- 5× more robust performance estimates
- Reduces variance from lucky/unlucky val splits
- Better hyperparameter selection

---

### Pattern 4: Multi-Run with Different Seeds (Experiment 5 Variant)

**Use Case**: Statistical evaluation with multiple independent train/val/test splits.

```python
from src.data import split_by_subjects_three_way

all_run_results = []

for run_idx in range(1, 31):  # 30 runs
    seed = base_seed + run_idx  # Different seed each run
    
    # Create unique train/val/test split for this run
    train, val, test = split_by_subjects_three_way(
        dataset,
        val_ratio=0.15,
        test_ratio=0.3,
        seed=seed,  # Different subjects each run!
        stratified=True
    )
    
    # Train and evaluate
    model = train_model(train, val)
    test_metrics = evaluate(model, test)
    all_run_results.append(test_metrics)

# Aggregate: mean ± std across 30 runs
mean_accuracy = np.mean([r['accuracy'] for r in all_run_results])
std_accuracy = np.std([r['accuracy'] for r in all_run_results])
```

**Advantages**:
- Robust statistical estimates (mean ± std)
- Tests sensitivity to train/test split
- Publishable results with confidence intervals

---

## Best Practices

### 1. Always Use Explicit Seeds

```python
# ✅ GOOD: Reproducible
train, test = split_by_subject_two_way(dataset, split_ratio=0.3, seed=42)

# ❌ BAD: Non-reproducible (different splits each run)
train, test = split_by_subject_two_way(dataset, split_ratio=0.3)
```

### 2. Enable Stratification for Final Experiments

```python
# ✅ GOOD: All classes guaranteed in all splits
train, test = split_by_subject_two_way(
    dataset, split_ratio=0.3, seed=42, stratified=True
)

# ⚠️ OK for prototyping, risky for final evaluation
train, test = split_by_subject_two_way(
    dataset, split_ratio=0.3, seed=42, stratified=False
)
```

### 3. Verify Splits Before Training

```python
from src.data import verify_subject_split_integrity

train, val, test = split_by_subjects_three_way(dataset, ...)

# Verify integrity
results = verify_subject_split_integrity(train, val, test, verbose=True)

# Assertions for production code
assert not results['has_subject_overlap'], "Data leakage detected!"
assert results['test_classes'] == results['total_classes'], "Missing classes in test!"
```

### 4. Never Modify Test Set

```python
# ❌ BAD: Test set contamination
test_augmented = augment(test_samples)  # DON'T augment test data

# ✅ GOOD: Only augment training
train_ds = build_dataset(train_samples, augment=True, shuffle=True)
test_ds = build_dataset(test_samples, augment=False, shuffle=False)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Sample-Level Splitting

```python
# ❌ WRONG: Splits samples, ignoring subjects
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(images, test_size=0.3, random_state=42)
# Result: Same subjects in train and test → data leakage
```

**Solution**: Always use subject-wise splitting functions.

### Pitfall 2: Forgetting Stratification

```python
# ⚠️ RISKY: Some classes may be missing from test
train, test = split_by_subject_two_way(dataset, split_ratio=0.3, seed=42)

# Later: test_confusion_matrix is 10×10 instead of 15×15
# Result: 5 exercises completely missing from test set
```

**Solution**: Enable `stratified=True` for final experiments.

### Pitfall 3: Not Freezing Test Set in CV

```python
# ❌ WRONG: Uses different test subjects in each fold
for fold in folds:
    train = all_other_folds
    test = fold
    # Problem: "test" subjects change each fold
```

**Solution**: Create frozen test set BEFORE building folds (Pattern 3 above).

### Pitfall 4: Augmenting Test Data

```python
# ❌ WRONG: Test set should not be augmented
test_ds = build_dataset(test_samples, augment=True, shuffle=True)
# Result: Inflated test accuracy
```

**Solution**: Always set `augment=False, shuffle=False` for test/val sets.

---

## Summary

### Key Takeaways

1. **Subject-wise splitting prevents data leakage** by ensuring subjects appear in only one split
2. **Stratification guarantees all classes in all splits**, critical for reliable evaluation
3. **Three splitting patterns** cover different experimental designs:
   - Two-way: Simple train/test
   - Three-way: Train/val/test with early stopping
   - K-fold: Robust cross-validation
4. **Verification utilities** detect data leakage before training
5. **Reproducibility** requires explicit seeds in all splitting functions

### Implementation Files

- Core logic: [src/data/data_loader.py](src/data/data_loader.py)
- Convenience wrappers: [src/data/dataset_builder.py](src/data/dataset_builder.py)
- Dataset loading: [src/data/data_loader.py](src/data/data_loader.py) (`load_front_side_geis()`, `load_pose_data()`)

### Next Steps

This splitting methodology is applied across all experiments. For specific experiment workflows, see:
- [EXPERIMENT_5_METHODOLOGY.md](EXPERIMENT_5_METHODOLOGY.md) - K-fold CV and multi-run experiments
- Individual experiment documentation in `src/scripts/experiment_*.py`
