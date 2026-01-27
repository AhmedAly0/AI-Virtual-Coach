# Pose Feature Engineering: Complete Documentation

## Overview

This document provides a comprehensive overview of the feature engineering approach developed for exercise recognition using MediaPipe 3D pose landmarks. It traces the evolution from basic angle-based features to advanced specialized discrimination features, culminating in a robust 37-feature vector optimized for 15-class exercise classification.

---

## Table of Contents

1. [MediaPipe Pose Extraction](#1-mediapipe-pose-extraction)
2. [Landmark Normalization](#2-landmark-normalization)
3. [Base Feature Set (19 Features)](#3-base-feature-set-19-features)
4. [View-Specific Feature Engineering](#4-view-specific-feature-engineering)
5. [Front View Specialized Features (18 Features)](#5-front-view-specialized-features-18-features)
6. [Side View Specialized Features (Planned)](#6-side-view-specialized-features-planned)
7. [Final Feature Vectors](#7-final-feature-vectors)
8. [Temporal Processing](#8-temporal-processing)
9. [NPZ File Structure](#9-npz-file-structure)
10. [Feature Type Reference](#10-feature-type-reference)

---

## 1. MediaPipe Pose Extraction

### Model Configuration

| Setting | Value |
|---------|-------|
| **Model** | `pose_landmarker_full.task` (~6MB) |
| **Output** | 33 body landmarks per frame |
| **Coordinates** | 3D (x, y, z) where z = depth from camera |
| **Detection Confidence** | 0.3 |
| **Tracking Confidence** | 0.3 |

### Key Landmark Indices

```
Upper Body:                    Lower Body:
├── 7, 8:   Left/Right Ear    ├── 23, 24: Left/Right Hip
├── 11, 12: Left/Right        ├── 25, 26: Left/Right Knee
│           Shoulder          ├── 27, 28: Left/Right Ankle
├── 13, 14: Left/Right Elbow  ├── 29, 30: Left/Right Heel
├── 15, 16: Left/Right Wrist  └── 31, 32: Left/Right Foot Index
├── 17, 18: Left/Right Pinky
├── 19, 20: Left/Right Index
└── 21, 22: Left/Right Thumb
```

---

## 2. Landmark Normalization

All landmarks undergo 3D normalization before feature computation, ensuring **scale and translation invariance**.

### Normalization Process

1. **Compute Pelvis Center (Origin)**
   $$pelvis = \frac{hip_{left} + hip_{right}}{2}$$

2. **Compute Mid-Shoulder Point**
   $$mid\_shoulder = \frac{shoulder_{left} + shoulder_{right}}{2}$$

3. **Compute Torso Length (Scale Factor)**
   $$torso\_length = \|mid\_shoulder - pelvis\|_2$$

4. **Normalize All 33 Landmarks**
   $$landmark_{norm} = \frac{landmark - pelvis}{torso\_length}$$

### Benefits
- **Camera distance invariant**: Features don't change with subject-camera distance
- **Body size invariant**: Different body proportions produce comparable features
- **Depth normalized**: Z-coordinate (depth) is also normalized

---

## 3. Base Feature Set (19 Features)

The base feature set consists of **13 joint angles** and **6 distance features**.

### 3.1 Joint Angles (13 Features)

Angles are computed using the 3D dot product formula:
$$\cos(\theta) = \frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}| \cdot |\vec{v_2}|}$$

| # | Feature Name | Description | Landmarks (a→b→c) |
|---|-------------|-------------|-------------------|
| 1 | `left_elbow` | Left arm flexion | shoulder→elbow→wrist |
| 2 | `right_elbow` | Right arm flexion | shoulder→elbow→wrist |
| 3 | `left_shoulder` | Left shoulder abduction | elbow→shoulder→hip |
| 4 | `right_shoulder` | Right shoulder abduction | elbow→shoulder→hip |
| 5 | `left_hip` | Left hip flexion | shoulder→hip→knee |
| 6 | `right_hip` | Right hip flexion | shoulder→hip→knee |
| 7 | `left_knee` | Left knee flexion | hip→knee→ankle |
| 8 | `right_knee` | Right knee flexion | hip→knee→ankle |
| 9 | `torso_lean` | Torso tilt from vertical | Computed from pelvis-shoulder axis |
| 10 | `left_ankle` | Left ankle dorsiflexion | knee→ankle→heel |
| 11 | `right_ankle` | Right ankle dorsiflexion | knee→ankle→heel |
| 12 | `left_wrist` | Left wrist angle | elbow→wrist→pinky |
| 13 | `right_wrist` | Right wrist angle | elbow→wrist→pinky |

### 3.2 Distance Features (6 Features)

| # | Feature Name | Description | Purpose |
|---|-------------|-------------|---------|
| 14 | `left_ear_shoulder_vert` | Vertical distance: ear to shoulder | Shrug detection |
| 15 | `right_ear_shoulder_vert` | Vertical distance: ear to shoulder | Shrug detection |
| 16 | `left_wrist_shoulder_dist` | Euclidean distance: wrist to shoulder | Arm extension |
| 17 | `right_wrist_shoulder_dist` | Euclidean distance: wrist to shoulder | Arm extension |
| 18 | `left_elbow_hip_dist` | Euclidean distance: elbow to hip | Arm tuck position |
| 19 | `right_elbow_hip_dist` | Euclidean distance: elbow to hip | Arm tuck position |

---

## 4. View-Specific Feature Engineering

A key insight from experimentation is that **different camera angles capture different aspects of exercise movements**. The same 33 MediaPipe landmarks are extracted from both views, and the same base features (19) are computed for both. However, **specialized discrimination features are view-specific**.

### 4.1 Why View-Specific Features?

| Aspect | Front View | Side View |
|--------|------------|----------|
| **Visibility** | Arm width, shoulder position, bilateral symmetry | Elbow flexion, torso lean, hip hinge depth |
| **Depth (Z-axis)** | Arms moving toward/away from camera | Lateral movement less visible |
| **Typical Performance** | ~75% accuracy (more confusion) | ~84% accuracy (clearer movements) |
| **Primary Confusions** | Curl variants, similar arm exercises | Different confusion patterns (TBD) |

### 4.2 Feature Engineering Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│  BASE FEATURES (19) - Shared across both views              │
│  ├── Joint Angles (13)                                      │
│  └── Distance Features (6)                                  │
├─────────────────────────────────────────────────────────────┤
│  SPECIALIZED FEATURES - View-specific                       │
│  ├── Front View: 18 features (implemented)                  │
│  │   └── Targets: Curl variants, Deadlift/Rows, etc.       │
│  └── Side View: TBD features (planned)                      │
│      └── May share some features with front view            │
└─────────────────────────────────────────────────────────────┘
```

**Note:** Some specialized features may be effective for both views (overlapping), while others are view-specific. The final feature sets will be determined by experimentation.

---

## 5. Front View Specialized Features (18 Features)

These features were developed to address **persistent confusion patterns** identified through multi-run experiments on the front view. Analysis of aggregated confusion matrices revealed four primary confusion clusters requiring targeted feature engineering.

### 5.1 Confusion Clusters Addressed (Front View)

| Cluster | Exercises Confused | Cross-Prediction Rate |
|---------|-------------------|----------------------|
| **Curl Variants** | Hammer ↔ EZ Bar ↔ Seated Biceps | 30-35% |
| **Hinge Movements** | Deadlift ↔ Rows | ~25% |
| **Arm Extensions** | Triceps Kickbacks ↔ Rows | ~22% |
| **Minimal Motion** | Shrugs ↔ Calf Raises | ~16% |

### 5.2 Group 1: Curl Discrimination (8 Features)

Designed to distinguish **Hammer Curls**, **EZ Bar Curls**, and **Seated Biceps Curls**.

| # | Feature Name | Description | Biomechanical Rationale |
|---|-------------|-------------|------------------------|
| 1 | `left_forearm_supination` | Forearm rotation estimate | EZ Bar ~30-45°, Hammer ~0°, Seated ~90° |
| 2 | `right_forearm_supination` | Forearm rotation estimate | Based on hand plane normal Y-component |
| 3 | `left_upper_arm_vertical` | Upper arm angle from vertical | Seated curls: arms behind torso |
| 4 | `right_upper_arm_vertical` | Upper arm angle from vertical | Incline bench changes arm position |
| 5 | `inter_wrist_distance` | Distance between wrists | EZ Bar: fixed bar width |
| 6 | `wrist_centerline_offset` | Wrist offset from body centerline | EZ Bar: wrists centered in front |
| 7 | `left_elbow_body_dist` | Left elbow from torso centerline | EZ Bar: elbows tucked |
| 8 | `right_elbow_body_dist` | Right elbow from torso centerline | Hammer: elbows can flare |

**Supination Computation:**
```
hand_normal = cross(wrist→index, wrist→thumb)
supination = arcsin(hand_normal_unit.y) × (180/π)
```

### 5.3 Group 2: Hinge Movement Discrimination (4 Features)

Designed to distinguish **Deadlift** from **Bent-Over Rows**.

| # | Feature Name | Description | Biomechanical Rationale |
|---|-------------|-------------|------------------------|
| 9 | `shoulder_width_ratio` | Shoulder width / hip width | Rows: shoulder retraction increases ratio |
| 10 | `left_wrist_hip_vertical` | Vertical distance: wrist to hip | Deadlift: bar rises; Rows: arms move |
| 11 | `right_wrist_hip_vertical` | Vertical distance: wrist to hip | Captures vertical bar/hand path |
| 12 | `hip_depth_ratio` | Hip height / knee height | Deadlift: hip rises; Rows: hip fixed |

**Key Insight:**
- **Deadlift**: Torso angle changes (rises from ~45° to vertical)
- **Rows**: Torso angle stays fixed at ~45° throughout

### 5.4 Group 3: Kickback Discrimination (2 Features)

Designed to distinguish **Triceps Kickbacks** from **Rows**.

| # | Feature Name | Description | Biomechanical Rationale |
|---|-------------|-------------|------------------------|
| 13 | `left_wrist_posterior` | Z-depth: wrist relative to hip | Kickback: wrist behind hip |
| 14 | `right_wrist_posterior` | Z-depth: wrist relative to hip | Rows: wrist beside hip |

**Computation:**
```
wrist_posterior = wrist.z - hip.z
```

### 5.5 Group 4: Elevation Discrimination (4 Features)

Designed to distinguish **Shrugs** from **Calf Raises**.

| # | Feature Name | Description | Biomechanical Rationale |
|---|-------------|-------------|------------------------|
| 15 | `left_heel_elevation` | Heel height vs foot index | Calf raises: heels rise |
| 16 | `right_heel_elevation` | Heel height vs foot index | Plantar flexion indicator |
| 17 | `shoulder_center_y` | Shoulder midpoint Y-position | Shrugs: shoulders rise |
| 18 | `ankle_center_y` | Ankle midpoint Y-position | Reference for lower body |

**Key Insight:**
- **Shrugs**: Joint angles barely change, but shoulders elevate
- **Calf Raises**: Joint angles barely change, but heels elevate

---

## 6. Side View Specialized Features (Planned)

The side view model currently uses only the **base 19 features** and achieves ~84% accuracy. View-specific specialized features for the side view are planned for future development.

### 6.1 Potential Confusion Clusters (Side View)

Based on preliminary analysis, the side view may have different confusion patterns:

| Cluster | Potential Confusions | Rationale |
|---------|---------------------|----------|
| **Pressing Movements** | Shoulder Press ↔ Inclined Press | Similar arm trajectory from side |
| **Hip Hinge Variants** | Deadlift ↔ Rows ↔ Kickbacks | Hip angle similar, arm movement differs |
| **Standing Exercises** | Calf Raises ↔ Shrugs ↔ Curls | Minimal lower body movement |

### 6.2 Planned Side View Features

Candidate features for side view discrimination (to be validated):

| Feature Category | Potential Features | Purpose |
|------------------|-------------------|--------|
| **Arm Trajectory** | Wrist path angle, elbow extension rate | Distinguish press vs curl movements |
| **Hip Mechanics** | Hip-knee-ankle alignment, torso angle | Better hinge movement discrimination |
| **Depth Cues** | Shoulder protraction/retraction | Row vs press distinction |

### 6.3 Feature Overlap Strategy

Some front view features may transfer well to side view:

| Feature | Front View Utility | Side View Utility |
|---------|-------------------|------------------|
| `shoulder_width_ratio` | ✅ Rows detection | ⚠️ Less visible from side |
| `wrist_hip_vertical` | ✅ Deadlift vs Rows | ✅ Likely useful |
| `heel_elevation` | ✅ Calf raises | ✅ Likely useful |
| `forearm_supination` | ✅ Curl variants | ❓ May not be visible |

**Note:** Final side view features will be determined through experimentation and confusion matrix analysis.

---

## 7. Final Feature Vectors

The feature vectors differ between views due to view-specific specialized features.

### 7.1 Front View Feature Vector (37 Features)

```
┌──────────────────────────────────────────────────────────────┐
│              FRONT VIEW FEATURE VECTOR (37)                  │
├──────────────────────────────────────────────────────────────┤
│  BASE FEATURES (19)                                          │
│  ├── Joint Angles (13): elbow, shoulder, hip, knee,         │
│  │                      torso, ankle, wrist (×2 sides)      │
│  └── Distances (6): ear-shoulder, wrist-shoulder,           │
│                     elbow-hip (×2 sides)                    │
├──────────────────────────────────────────────────────────────┤
│  FRONT-SPECIFIC SPECIALIZED FEATURES (18)                    │
│  ├── Curl Group (8): supination, upper arm vertical,        │
│  │                   inter-wrist, centerline, elbow-body    │
│  ├── Hinge Group (4): shoulder ratio, wrist-hip vert,       │
│  │                    hip depth                              │
│  ├── Kickback Group (2): wrist posterior                    │
│  └── Elevation Group (4): heel elevation, shoulder/ankle Y  │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Side View Feature Vector (19 Features - Current)

```
┌──────────────────────────────────────────────────────────────┐
│              SIDE VIEW FEATURE VECTOR (19)                   │
├──────────────────────────────────────────────────────────────┤
│  BASE FEATURES (19)                                          │
│  ├── Joint Angles (13): elbow, shoulder, hip, knee,         │
│  │                      torso, ankle, wrist (×2 sides)      │
│  └── Distances (6): ear-shoulder, wrist-shoulder,           │
│                     elbow-hip (×2 sides)                    │
├──────────────────────────────────────────────────────────────┤
│  SIDE-SPECIFIC SPECIALIZED FEATURES (TBD)                    │
│  └── To be developed based on side view confusion analysis  │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 Feature Count Summary

| Feature Set | Front View | Side View | Description |
|-------------|------------|-----------|-------------|
| `angles` | 13 | 13 | Joint angles only |
| `distances` | 6 | 6 | Distance features only |
| `all` | 19 | 19 | Base set (angles + distances) |
| `specialized` | 18 | TBD | View-specific discrimination features |
| `all_extended` | 37 | 19 (current) | Full feature set per view |

---

## 8. Temporal Processing

### 8.1 Temporal Resampling

Videos are resampled to a fixed length (`T_fixed = 50` frames) using linear interpolation:

```
Original:  [frame_0, frame_1, ..., frame_T_orig]
           ↓ Linear interpolation
Resampled: [frame_0, frame_1, ..., frame_49]
```

**Edge Cases:**
- `T_orig = 0`: Fill with zeros
- `T_orig = 1`: Replicate single frame
- `T_orig > 1`: Linear interpolation per feature

### 8.2 Tempo Preservation

Since temporal resampling destroys timing information (fast vs slow reps look identical after resampling), **tempo metadata** is preserved separately:

| Tempo Feature | Description | Computation |
|---------------|-------------|-------------|
| `tempo_duration_sec` | FPS-normalized duration | `total_frames / fps` |
| `tempo_frame_count` | Raw frame count | All video frames |
| `tempo_fps` | Original video FPS | Video metadata |

**Usage:** Tempo features can be concatenated with pose features for models that need timing information:
```python
X_with_tempo = np.concatenate([
    X_temporal.reshape(N, -1),           # (N, 50×37)
    tempo_duration[:, np.newaxis]        # (N, 1)
], axis=1)                               # (N, 1851)
```

---

## 9. NPZ File Structure

### Output File Naming

```
datasets/Mediapipe pose estimates/
├── pose_data_front.npz    # Front view with base + front-specific features
├── pose_data_side.npz     # Side view with base features (+ side-specific TBD)
```

### NPZ Contents

```python
{
    # Landmark data
    'X_landmarks':      (N, T, 33, 3),    # Raw normalized landmarks
    
    # Base features
    'X_angles':         (N, T, 13),       # Joint angles
    'X_distances':      (N, T, 6),        # Distance features
    'X_all_features':   (N, T, 19),       # Angles + distances
    
    # Specialized features
    'X_specialized':    (N, T, 18),       # Discrimination features
    
    # Metadata
    'exercise_names':   (N,),             # Exercise labels
    'subject_ids':      (N,),             # Subject IDs
    'tempo_fps':        (N,),             # FPS per sample
    'tempo_duration_sec': (N,),           # Duration in seconds
    'tempo_frame_count': (N,),            # Original frame count
    'view':             str,              # 'front' or 'side'
    'T_fixed':          int,              # Fixed sequence length (50)
    'angle_names':      list[str],        # 13 angle names
    'distance_names':   list[str],        # 6 distance names
    'specialized_names': list[str],       # 18 specialized names
}
```

---

## 10. Feature Type Reference

### Available Feature Types in Data Loader

| Feature Type | Features Loaded | Flattened Dimension |
|--------------|-----------------|---------------------|
| `'angles'` | 13 joint angles | 13 × 50 = 650 |
| `'distances'` | 6 distance features | 6 × 50 = 300 |
| `'all'` | 19 base features | 19 × 50 = 950 |
| `'specialized'` | 18 discrimination features | 18 × 50 = 900 |
| `'all_extended'` | 37 full features | 37 × 50 = 1,850 |
| `'base_specialized'` | Same as `'all_extended'` | 37 × 50 = 1,850 |

### Loading Examples

```python
from src.data.data_loader import load_pose_npz_temporal

# Load front view base features (19)
X, y, subject_ids, meta = load_pose_npz_temporal(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front.npz',
    feature_type='all'
)
# X.shape: (N, 950)  # Flattened 19 × 50

# Load front view full features (37)
X, y, subject_ids, meta = load_pose_npz_temporal(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front.npz',
    feature_type='all_extended'
)
# X.shape: (N, 1850)  # Flattened 37 × 50

# Load side view features (19 base, specialized TBD)
X, y, subject_ids, meta = load_pose_npz_temporal(
    npz_path='datasets/Mediapipe pose estimates/pose_data_side.npz',
    feature_type='all'
)
# X.shape: (N, 950)  # Flattened 19 × 50
```

---

## Summary

### Feature Engineering Evolution

| Phase | Features | Key Addition |
|-------|----------|--------------|
| **Initial** | 9 angles | Core joint angles |
| **Phase 1** | 19 (13 angles + 6 distances) | Ankle, wrist angles; ear-shoulder, arm distances |
| **Phase 2** | 37 (19 base + 18 specialized) | Curl, hinge, kickback, elevation discrimination |

### Key Design Principles

1. **3D Coordinates**: Leverage MediaPipe's depth (z) for movements toward/away from camera
2. **Normalization**: Pelvis-centered, torso-length normalized for body size invariance
3. **Targeted Features**: Biomechanically-motivated features for specific confusion patterns
4. **Tempo Preservation**: Separate duration metadata to preserve timing after resampling

### Implementation Reference

| Component | File |
|-----------|------|
| Feature extraction | `src/preprocessing/preprocess_pose_RGB.py` |
| Data loading | `src/data/data_loader.py` |
| Preprocessing notebook | `notebooks/exer_recog/00_pose_preprocessing.ipynb` |
| MLP training | `notebooks/exer_recog/06_pose_mlp.ipynb` |

---

*Last Updated: January 2026*
