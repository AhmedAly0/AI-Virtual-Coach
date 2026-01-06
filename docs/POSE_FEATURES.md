# Pose Feature Extraction Documentation

## Overview

This document describes the two types of pose estimate feature vectors extracted from exercise videos using MediaPipe **3D pose landmarks**: **static features** (aggregated statistics) and **temporal features** (time-series sequences).

## MediaPipe Landmark Extraction

- **Model**: MediaPipe Pose Landmarker (full, float16)
- **Input**: Exercise videos (front and side views)
- **Output**: 33 body landmarks per frame with **3D coordinates (x, y, z)**
- **Detection thresholds**: 0.3 (confidence and tracking)
- **Depth information**: z-coordinate represents depth from camera plane

## Joint Angle Computation

Nine normalized joint angles are computed from MediaPipe 3D landmarks using the `calculate_angle()` function with **3D dot product formula**:

**Angle Formula**: `cos(θ) = (v1·v2) / (|v1||v2|)` where:
- v1 = vector from joint b to point a (in 3D space)
- v2 = vector from joint b to point c (in 3D space)

| Angle Name | Description | MediaPipe Landmarks (a-b-c) |
|------------|-------------|----------------------------|
| `left_elbow` | Left arm flexion (3D) | 11 → 13 → 15 |
| `right_elbow` | Right arm flexion (3D) | 12 → 14 → 16 |
| `left_shoulder` | Left shoulder angle (3D) | 13 → 11 → 23 |
| `right_shoulder` | Right shoulder angle (3D) | 14 → 12 → 24 |
| `left_hip` | Left hip angle (3D) | 11 → 23 → 25 |
| `right_hip` | Right hip angle (3D) | 12 → 24 → 26 |
| `left_knee` | Left knee flexion (3D) | 23 → 25 → 27 |
| `right_knee` | Right knee flexion (3D) | 24 → 26 → 28 |
| `torso_lean` | Torso tilt from vertical (3D) | Computed from mid-shoulder and pelvis in 3D |

**Angle calculation**: 3D dot product-based computation returning degrees (0-180°).

## Landmark Normalization

Before angle computation, landmarks are normalized using `_normalize_landmarks()` for scale and translation invariance **in 3D space**:

1. **Pelvis center in 3D**: Midpoint of left hip (23) and right hip (24)
   - $pelvis_x = \frac{x_{23} + x_{24}}{2}$
   - $pelvis_y = \frac{y_{23} + y_{24}}{2}$
   - $pelvis_z = \frac{z_{23} + z_{24}}{2}$

2. **Mid-shoulder point in 3D**: Midpoint of left shoulder (11) and right shoulder (12)
   - $mid\\_shoulder_x = \frac{x_{11} + x_{12}}{2}$
   - $mid\\_shoulder_y = \frac{y_{11} + y_{12}}{2}$
   - $mid\\_shoulder_z = \frac{z_{11} + z_{12}}{2}$

3. **Torso length in 3D** (normalization factor):
   - $torso\\_length = \sqrt{(mid\\_shoulder_x - pelvis_x)^2 + (mid\\_shoulder_y - pelvis_y)^2 + (mid\\_shoulder_z - pelvis_z)^2}$

4. **Normalization in 3D**:
   - $x_{norm} = \frac{x - pelvis_x}{torso\\_length}$
   - $y_{norm} = \frac{y - pelvis_y}{torso\\_length}$
   - $z_{norm} = \frac{z - pelvis_z}{torso\\_length}$

**Purpose**: Makes features invariant to camera distance, subject body size, and depth variations.

**Key Advantage of 3D**: Captures depth information for movements toward/away from camera (e.g., forward lean in squats, arm extension in presses).

## Feature Type 1: Static Features

**Function**: `build_static_rep_features()`

**Method**: Aggregates temporal statistics for each angle across the entire repetition.

**Statistics per angle** (5 metrics):
- Mean
- Standard deviation
- Minimum
- Maximum  
- Range (max - min)

**Dimensionality**: `5 statistics × 9 angles = 45 features`

**Output shape**: `(num_reps, 45)`

**Data type**: `float32`

**Use case**: Classification models that don't require temporal information (e.g., MLP, SVM, Random Forest).

## Feature Type 2: Temporal Features

**Function**: `build_temporal_rep_features()`

**Method**: Resamples each repetition to fixed length using linear interpolation.

**Parameters**:
- `T_fixed = 50`: Target number of timesteps (configured in notebook)
- Interpolation: Scipy linear interpolation applied per angle independently

**Resampling logic**:
- Original time axis: `np.linspace(0, 1, T_orig)`
- Target time axis: `np.linspace(0, 1, T_fixed)`
- Edge cases:
  - `T_orig = 0`: Fill with zeros
  - `T_orig = 1`: Replicate single frame
  - `T_orig > 1`: Linear interpolation

**Output shape**: `(num_reps, 50, 9)`

**Data type**: `float32`

**Use case**: Temporal models (LSTM, GRU, Conv1D, Transformers) that leverage sequence information.

## Output Files

### File Naming Convention

```
datasets/Mediapipe pose estimates/
├── pose_data_front_static.npz
├── pose_data_front_temporal.npz
├── pose_data_side_static.npz
└── pose_data_side_temporal.npz
```

### Static NPZ Contents

```python
{
    'X_static': ndarray(num_reps, 45),          # Static features
    'exercise_names': ndarray(num_reps, object), # Exercise labels
    'subject_ids': ndarray(num_reps, int32),     # Subject IDs
    'view': str,                                  # 'front' or 'side'
    'angle_names': list[str],                     # 9 angle names
}
```

### Temporal NPZ Contents

```python
{
    'X_temporal': ndarray(num_reps, 50, 9),      # Temporal features
    'exercise_names': ndarray(num_reps, object),  # Exercise labels
    'subject_ids': ndarray(num_reps, int32),      # Subject IDs
    'view': str,                                   # 'front' or 'side'
    'T_fixed': int,                                # Fixed length (50)
    'angle_names': list[str],                      # 9 angle names
}
```

### Angle Names Array

```python
['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'torso_lean']
```

## Processing Pipeline

**Main function**: `extract_pose_estimates(clips_path, view, T_fixed, output_path)`

**Workflow**:
1. Scan directory for video files matching view
2. Extract pose landmarks per frame using MediaPipe Tasks API
3. Normalize landmarks and compute 9 angles per frame
4. Aggregate into static features (statistics)
5. Resample into temporal features (fixed length)
6. Save separate NPZ files for static and temporal

**Views processed**: `front`, `side`

**Per-video instance**: Fresh `PoseLandmarker` created to avoid timestamp contamination in VIDEO mode.

## Implementation Notes

- **Timestamp handling**: Frame index used as timestamp (monotonically increasing)
- **Failed normalization**: Frames with invalid torso length (<1e-6) are skipped
- **Video reading**: OpenCV (`cv2.VideoCapture`)
- **Color conversion**: BGR → RGB for MediaPipe compatibility
- **Subject ID extraction**: Regex-based parsing from folder names (e.g., "Volunteer #3" → 3)

## References

- **Preprocessing script**: `src/preprocessing/preprocess_pose_RGB.py`
- **Notebook**: `notebooks/exer_recog/00_pose_preprocessing.ipynb`
- **Model download**: [MediaPipe Pose Landmarker Lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)
