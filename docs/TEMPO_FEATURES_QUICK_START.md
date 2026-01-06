# Quick Start: Using Tempo Features

## What Changed?

Your NPZ files now include **tempo preservation features** that capture exercise speed/timing:

```python
# OLD NPZ structure (before)
{
    'X_temporal': (N, 50, 9),  # Resampled to 50 frames
    'exercise_names': (N,),
    'subject_ids': (N,),
}

# NEW NPZ structure (after)
{
    'X_temporal': (N, 50, 9),         # Still resampled to 50
    'exercise_names': (N,),
    'subject_ids': (N,),
    'tempo_duration_sec': (N,),       # ⭐ NEW: Duration in seconds
    'tempo_frame_count': (N,),        # ⭐ NEW: Original frame count
    'tempo_fps': (N,),                # ⭐ NEW: Video FPS
}
```

## Quick Usage Examples

### 1. Load Data with Tempo

```python
import numpy as np

# Load temporal features
data = np.load('datasets/Mediapipe pose estimates/pose_data_front_temporal.npz', 
               allow_pickle=True)

X_temporal = data['X_temporal']            # (1574, 50, 9)
tempo_duration = data['tempo_duration_sec'] # (1574,) - seconds
tempo_frames = data['tempo_frame_count']   # (1574,) - frame count
tempo_fps = data['tempo_fps']              # (1574,) - FPS
exercise_names = data['exercise_names']    # (1574,)
subject_ids = data['subject_ids']          # (1574,)
```

### 2. Use Tempo as a Feature

#### Option A: Append to Static Features
```python
# Load static features
data_static = np.load('pose_data_front_static.npz', allow_pickle=True)
X_static = data_static['X_static']  # (1574, 45)
tempo_duration = data_static['tempo_duration_sec']  # (1574,)

# Concatenate tempo as additional feature
X_static_with_tempo = np.concatenate([
    X_static,
    tempo_duration[:, np.newaxis]  # Add as column
], axis=1)  # Shape: (1574, 46)
```

#### Option B: Append to Temporal Features (Flattened)
```python
# Flatten temporal + add tempo
X_temporal_flat = X_temporal.reshape(len(X_temporal), -1)  # (1574, 450)
X_with_tempo = np.concatenate([
    X_temporal_flat,
    tempo_duration[:, np.newaxis]
], axis=1)  # Shape: (1574, 451)
```

#### Option C: Use Multiple Tempo Features
```python
# Use all three tempo features
tempo_features = np.column_stack([
    tempo_duration,      # Duration in seconds
    tempo_frames,        # Frame count
    tempo_fps           # FPS
])  # Shape: (1574, 3)

X_combined = np.concatenate([
    X_static,
    tempo_features
], axis=1)  # Shape: (1574, 48)
```

### 3. Analyze Tempo Distribution

```python
import matplotlib.pyplot as plt

# Plot tempo distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(tempo_duration, bins=30, edgecolor='black')
plt.xlabel('Duration (seconds)')
plt.title('Rep Duration Distribution')

plt.subplot(1, 3, 2)
plt.hist(tempo_frames, bins=30, edgecolor='black')
plt.xlabel('Frame Count')
plt.title('Frame Count Distribution')

plt.subplot(1, 3, 3)
plt.bar(*np.unique(tempo_fps, return_counts=True))
plt.xlabel('FPS')
plt.ylabel('Count')
plt.title('FPS Distribution')

plt.tight_layout()
plt.show()

# Statistics
print(f"Duration: {tempo_duration.min():.2f}s - {tempo_duration.max():.2f}s")
print(f"Median: {np.median(tempo_duration):.2f}s")
print(f"FPS values: {np.unique(tempo_fps)}")
```

### 4. Filter by Tempo

```python
# Find slow reps (>3 seconds)
slow_reps = tempo_duration > 3.0
X_slow = X_temporal[slow_reps]
print(f"Slow reps: {slow_reps.sum()} / {len(tempo_duration)}")

# Find fast reps (<1.5 seconds)
fast_reps = tempo_duration < 1.5
X_fast = X_temporal[fast_reps]
print(f"Fast reps: {fast_reps.sum()} / {len(tempo_duration)}")

# Per-exercise tempo analysis
import pandas as pd

df = pd.DataFrame({
    'exercise': exercise_names,
    'duration': tempo_duration
})

tempo_by_exercise = df.groupby('exercise')['duration'].describe()
print(tempo_by_exercise)
```

### 5. Use in Model Training (Example)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare features with tempo
X = np.concatenate([
    X_static,
    tempo_duration[:, np.newaxis]
], axis=1)

y = exercise_names  # or your labels

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Feature importance
feature_names = [f"static_{i}" for i in range(45)] + ['tempo_duration']
importances = pd.Series(clf.feature_importances_, index=feature_names)
print("\nTop features including tempo:")
print(importances.nlargest(10))
```

## Tempo Feature Meanings

| Feature | Description | Use Case |
|---------|-------------|----------|
| `tempo_duration_sec` | Duration in seconds (FPS-normalized) | **Primary tempo feature** - use for tempo-based comparisons |
| `tempo_frame_count` | Total frames in original video | Optional - frame count before resampling |
| `tempo_fps` | Original video FPS | Optional - for debugging or reconstruction |

## Key Insights

✅ **tempo_duration_sec is FPS-independent**
- A 2-second rep at 30fps (60 frames) has the same tempo_duration_sec as a 2-second rep at 60fps (120 frames)
- Safe to compare across videos with different FPS

✅ **Tempo preserved despite resampling**
- X_temporal is resampled to 50 frames (loses timing)
- tempo_duration_sec preserves the original speed
- Best of both worlds: fixed-length sequences + tempo information

✅ **Use for exercise assessment**
- Compare rep speed against reference
- Detect too-fast or too-slow execution
- Important for exercise quality evaluation

## Common Questions

**Q: Do I need to re-extract pose data?**
A: Only if you want the tempo features. Existing NPZ files will work but won't have tempo arrays.

**Q: Will old code break?**
A: No! The new keys are additions. Old code that doesn't use tempo features will work as before.

**Q: Which tempo feature should I use?**
A: Use `tempo_duration_sec` - it's FPS-normalized and directly interpretable.

**Q: Can I ignore tempo features?**
A: Yes, but you'll miss important information about exercise execution speed.
