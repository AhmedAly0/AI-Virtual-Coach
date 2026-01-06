# Pose Estimation Upgrades - Implementation Summary

## Overview

This document summarizes the comprehensive upgrades made to the pose estimation pipeline, addressing model quality, tempo preservation, and optimal feature extraction parameters.

## 1. Upgraded to MediaPipe Full Model

### Changes Made
- **Model Path Updated**: Changed from `pose_landmarker_lite.task` to `pose_landmarker_full.task`
- **Download URL Updated**: Now downloads the full model (~6MB vs ~3MB lite)
- **Expected Benefits**: Better accuracy in pose landmark detection, especially for complex exercises

### Files Modified
- [src/preprocessing/preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py)
  - Updated `MODEL_PATH` constant
  - Updated error messages with correct download URL
- [notebooks/exer_recog/00_pose_preprocessing.ipynb](notebooks/exer_recog/00_pose_preprocessing.ipynb)
  - Updated model download cell

### Action Required
**You need to download the new model file:**
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
  -O datasets/pose_landmarker_full.task
```

Or run the first cells of the preprocessing notebook which will download it automatically.

---

## 2. Tempo Preservation with FPS Normalization

### The Problem
Temporal feature extraction was resampling all videos to a fixed 50 frames, which **destroyed tempo information**:
- Fast rep (30 frames, 1 sec) → 50 frames
- Slow rep (90 frames, 3 sec) → 50 frames
- Both looked identical after resampling, losing the speed/tempo signal

### The Solution
Added **tempo as separate feature arrays** that preserve timing information:

1. **tempo_duration_sec**: FPS-normalized duration in seconds
   - Formula: `total_frames / fps`
   - Example: 60 frames at 30fps = 2.0 seconds
   - This is FPS-independent and directly comparable

2. **tempo_frame_count**: Raw frame count (all video frames)
   - Total frames in the original video
   - Includes frames where pose detection may have failed

3. **tempo_fps**: Original video FPS
   - Allows reconstruction: `duration = frame_count / fps`
   - Handles videos with different FPS (30fps vs 60fps)

### Implementation Details

#### Modified Functions

**`extract_features_from_video()`**
- **Before**: Returned only `np.ndarray` of features
- **After**: Returns tuple `(features, total_frames, fps)`
- Extracts FPS from video metadata
- Counts ALL frames (not just valid pose frames)

**`process_video_list()`**
- **Before**: Returned only list of sequences
- **After**: Returns `(sequences, frame_counts, fps_values)`
- Collects tempo metadata for all videos

**`extract_pose_estimates()`**
- Collects tempo arrays: `all_tempo_frame_counts`, `all_tempo_fps`
- Computes FPS-normalized duration: `tempo_duration_sec = frame_counts / fps`
- Adds tempo arrays to dataset dictionary
- Saves tempo features in both static and temporal NPZ files

### Files Modified
- [src/preprocessing/preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py)
  - Updated function signatures and return types
  - Added tempo feature computation
  - Updated NPZ save operations
- [notebooks/exer_recog/00_pose_preprocessing.ipynb](notebooks/exer_recog/00_pose_preprocessing.ipynb)
  - Updated summary displays to show tempo statistics

### NPZ File Structure (New)

**Static NPZ** (`pose_data_*_static.npz`):
```python
{
    'X_static': (num_reps, 45),          # Statistical features
    'exercise_names': (num_reps,),        # Exercise labels
    'subject_ids': (num_reps,),           # Subject IDs
    'tempo_duration_sec': (num_reps,),    # ⭐ Duration in seconds (FPS-normalized)
    'tempo_frame_count': (num_reps,),     # ⭐ Total frames
    'tempo_fps': (num_reps,),             # ⭐ FPS values
    'view': str,
    'angle_names': list[str],
}
```

**Temporal NPZ** (`pose_data_*_temporal.npz`):
```python
{
    'X_temporal': (num_reps, T_fixed, 9), # Resampled sequences
    'exercise_names': (num_reps,),         # Exercise labels
    'subject_ids': (num_reps,),            # Subject IDs
    'tempo_duration_sec': (num_reps,),     # ⭐ Duration in seconds
    'tempo_frame_count': (num_reps,),      # ⭐ Total frames
    'tempo_fps': (num_reps,),              # ⭐ FPS values
    'view': str,
    'T_fixed': int,
    'angle_names': list[str],
}
```

### Usage Example
```python
# Load temporal features
data = np.load('pose_data_front_temporal.npz', allow_pickle=True)

X_temporal = data['X_temporal']           # (N, 50, 9) - time-normalized
tempo_duration = data['tempo_duration_sec'] # (N,) - actual duration in seconds
tempo_fps = data['tempo_fps']              # (N,) - original FPS

# Now you can use tempo as a feature for your assessment model!
# Example: Concatenate tempo to your feature vector
X_with_tempo = np.concatenate([
    X_temporal.reshape(len(X_temporal), -1),  # Flatten temporal
    tempo_duration[:, np.newaxis]              # Add tempo feature
], axis=1)
```

---

## 3. Frame Distribution Analysis Tools

### Purpose
Determine the optimal `T_fixed` value based on actual video frame distributions, rather than using an arbitrary value of 50.

### New Files Created

#### A. Analysis Script: `src/preprocessing/analyze_frame_distribution.py`

**Features:**
- Scans all videos in `Clips/` directory
- Extracts metadata: frame count, FPS, duration, resolution
- Computes comprehensive statistics (min, max, mean, median, percentiles)
- Generates visualizations:
  - Histograms of frame count distribution
  - Box plots per exercise
  - Heatmaps of median frame counts
  - Violin plots comparing views
- Exports results:
  - CSV with per-video metadata
  - JSON with statistics summary

**Usage:**
```bash
cd src/preprocessing

# Analyze front view only
python analyze_frame_distribution.py --view front

# Analyze both views
python analyze_frame_distribution.py --view both

# Custom paths
python analyze_frame_distribution.py \
  --clips_path ../../datasets/Clips \
  --output_dir ../../plots/frame_distribution_analysis \
  --view both
```

**Outputs:**
- `plots/frame_distribution_analysis/histogram_*.png`
- `plots/frame_distribution_analysis/boxplot_per_exercise_*.png`
- `plots/frame_distribution_analysis/heatmap_*.png`
- `plots/frame_distribution_analysis/frame_metadata_*.csv`
- `plots/frame_distribution_analysis/frame_statistics_*.json`

#### B. Interactive Notebook: `notebooks/exer_recog/00b_frame_distribution_analysis.ipynb`

**Features:**
- Interactive exploration of frame distributions
- Per-exercise breakdown
- FPS variation analysis
- T_fixed optimization recommendations
- Front vs Side view comparison
- Visual comparison of different T_fixed values (32, 50, 64, 80, 100)

**Key Outputs:**
- Statistical summary tables
- Distribution plots
- Recommendations for optimal T_fixed
- Impact analysis of different T_fixed values

---

## 4. Workflow for Re-running Pose Extraction

### Step 1: Download New Model
Run the first cells in [notebooks/exer_recog/00_pose_preprocessing.ipynb](notebooks/exer_recog/00_pose_preprocessing.ipynb) or:
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
  -O datasets/pose_landmarker_full.task
```

### Step 2: Analyze Frame Distribution (Optional but Recommended)
```bash
cd src/preprocessing
python analyze_frame_distribution.py --view both
```

Or run [notebooks/exer_recog/00b_frame_distribution_analysis.ipynb](notebooks/exer_recog/00b_frame_distribution_analysis.ipynb)

This will show you:
- Median frame count per exercise
- Recommended T_fixed value
- FPS variations

### Step 3: Re-run Pose Extraction
Open and run [notebooks/exer_recog/00_pose_preprocessing.ipynb](notebooks/exer_recog/00_pose_preprocessing.ipynb)

This will:
1. Use the new full model (better accuracy)
2. Extract pose features as before
3. **Add tempo features** to the NPZ files

### Step 4: Verify Tempo Features
After extraction, check the output:
```python
data = np.load('datasets/Mediapipe pose estimates/pose_data_front_temporal.npz', 
               allow_pickle=True)

print("Keys:", list(data.keys()))
print("Tempo duration shape:", data['tempo_duration_sec'].shape)
print("Tempo duration range:", data['tempo_duration_sec'].min(), 
      "-", data['tempo_duration_sec'].max(), "seconds")
print("FPS values:", np.unique(data['tempo_fps']))
```

---

## 5. Next Steps & Recommendations

### Immediate Actions
1. ✅ **Download MediaPipe Full model** (run notebook cell or wget)
2. ✅ **Run frame distribution analysis** to understand your data
3. ✅ **Re-run pose extraction** with new code (if you want tempo features)

### For Your Assessment Model
When building your exercise assessment model, you can now use tempo as a feature:

```python
# Example: Load data with tempo
data = np.load('pose_data_front_temporal.npz', allow_pickle=True)

X_temporal = data['X_temporal']           # (N, 50, 9)
tempo_duration = data['tempo_duration_sec'] # (N,) - ⭐ NEW!

# Option 1: Use tempo as an additional feature
X_combined = np.concatenate([
    X_temporal.reshape(len(X_temporal), -1),
    tempo_duration[:, np.newaxis]
], axis=1)

# Option 2: Use tempo for weighted scoring
# Fast exercises (low duration) vs slow exercises (high duration)
# can be compared fairly
```

### T_fixed Optimization
Based on frame analysis results:
- If median frame count is close to 50: **Keep T_fixed = 50** ✅
- If median is significantly different: **Consider adjusting T_fixed**
- Test multiple values: `[32, 50, 64, 80, 100]` and evaluate model performance

### Future Enhancements
- **3D Pose Estimation**: MediaPipe provides `.z` coordinates (depth) which are currently ignored
- **Pose Confidence Filtering**: Use `landmark.visibility` to filter low-confidence detections
- **Variable-Length Sequences**: Instead of resampling, use LSTM with masking for true temporal modeling

---

## Summary

| Feature | Before | After |
|---------|--------|-------|
| **Model** | Lite (~3MB) | Full (~6MB) |
| **Accuracy** | Basic | Improved |
| **Tempo Preservation** | ❌ Lost in resampling | ✅ Separate FPS-normalized features |
| **T_fixed Selection** | Arbitrary (50) | Data-driven analysis |
| **NPZ Structure** | 4 keys | 7 keys (+ tempo arrays) |
| **Analysis Tools** | None | Script + Notebook |

## Questions or Issues?

If you encounter any problems:
1. Check that the new model file exists: `datasets/pose_landmarker_full.task`
2. Verify NPZ files have the new keys: `'tempo_duration_sec'`, `'tempo_frame_count'`, `'tempo_fps'`
3. Run the frame analysis notebook to understand your data distribution

All changes are backward compatible with your existing experiment code (the new tempo features are simply additional arrays in the NPZ files).
