# AI Virtual Coach — Pose-Based Exercise Recognition, Quality Assessment & Coaching Feedback

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Dataset](https://img.shields.io/badge/dataset-51%20volunteers%2C%2015%20exercises-orange)

**Lightweight pose-based AI system for real-time exercise recognition, form assessment, and personalized coaching feedback from smartphone video.**

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Technical Deep Dives](#technical-deep-dives)
  - [Pose Estimation & Feature Engineering](#pose-estimation--feature-engineering)
  - [Exercise Recognition Module](#exercise-recognition-module)
  - [Subject-Wise Splitting Methodology](#subject-wise-splitting-methodology)
  - [Exercise Assessment Module](#exercise-assessment-module)
  - [Coaching Agent (LangGraph + Gemini)](#coaching-agent-langgraph--gemini)
  - [Backend API Contract](#backend-api-contract)
- [Configuration Reference](#configuration-reference)
- [Experimental Results](#experimental-results)
- [Supported Exercises](#supported-exercises)
- [Key Dependencies](#key-dependencies)
- [Testing](#testing)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Authors / Contact](#authors--contact)

---

## Overview

The **AI Virtual Coach** is a comprehensive system for analyzing resistance-training exercise videos and providing real-time, personalized coaching feedback. It solves a practical problem: most people training without a coach never receive feedback on their form, leading to ineffective training and injury risk.

### Core Capabilities

1. **Exercise Recognition** — Identifies which of 15 resistance exercises is being performed (Dumbbell Shoulder Press, Squats, Deadlifts, etc.)
2. **Form Assessment** — Scores each repetition across 5 biomechanical aspects (0–10 scale) in real time
3. **Coaching Agent** — Generates natural language feedback using LLM, identifying fatigue patterns, detecting form breakdown, and providing actionable tips

### Key Features

- **Smartphone-friendly** — Works with standard 1080×1920 video from consumer phones
- **Subject-independent** — Generalizes to new users not seen during training
- **Multi-aspect scoring** — Evaluates form across 5 criteria per exercise (not just overall score)
- **Fatigue detection** — Identifies when form degrades due to fatigue and suggests intervention
- **Explainable** — Every recommendation traces back to specific reps and metrics
- **Fast inference** — 30 seconds of video processed in <30 seconds on typical hardware

### Numbers

- **Dataset**: 51 volunteers, 15 exercises, 308 videos (front + side views)
- **Recognition accuracy**: 90.36% macro F1 (side view), 86.94% macro F1 (front view) — subject-disjoint, 30-run evaluation
- **Assessment accuracy**: ~3.5 MAE on 0–10 scale
- **Feature representation**: 37 biomechanical features per frame (base 19 + view-specific 18)
- **Deployment**: Subject-disjoint protocol ensures generalization to new users

---

## System Architecture

The AI Virtual Coach processes video through a **4-stage pipeline**:

```
RGB Video (Front/Side View)
    │
    ├─► STAGE 1: Pose Estimation & Feature Engineering
    │       └─► MediaPipe Pose Landmarker (33 landmarks)
    │       └─► 3D Normalization (pelvis-centered, torso-length scaled)
    │       └─► 37 Biomechanical Features (13 angles + 6 distances + 18 specialized)
    │
    ├─► STAGE 2: Exercise Recognition
    │       └─► Feedforward NN Classifier [512, 256, 128]
    │       └─► Input: (50, 37) temporal features
    │       └─► Output: Exercise class (1–15) + confidence
    │
    ├─► STAGE 3: Per-Repetition Assessment
    │       └─► Rep Segmentation (exercise/view-specific signal)
    │       └─► Temporal CNN Regression (15 exercise-specific models)
    │       └─► Output: 5 aspect scores (0–10) × N reps
    │
    └─► STAGE 4: Coaching Agent (LangGraph)
            ├─► Load Exercise Criteria (rule-based)
            ├─► Analyze Aggregated Scores (warnings if low)
            ├─► Analyze Rep Trends (fatigue, consistency, improvement)
            ├─► Generate LLM Feedback (Gemini 2.5 Flash)
            └─► Format Response (structured JSON + natural language)
```

### Pipeline Stages Summary

| Stage | Component | Input | Output | Key Technology |
|-------|-----------|-------|--------|----------------|
| **1** | Pose Estimation & Feature Extraction | RGB Video (frames) | (50, 37) temporal features | MediaPipe Pose Landmarker |
| **2** | Exercise Recognition | Temporal features | Exercise class (1–15) + confidence | Feedforward NN |
| **3** | Per-Rep Assessment | Rep sequences | 5 aspect scores (0–10) × N reps | Temporal CNN (×15 models) |
| **4** | Coaching Agent | Per-rep scores + criteria | Aggregated feedback, trends, NL advice | LangGraph + Gemini 2.5 Flash |

### Stage-by-Stage Details

**Stage 1 — Pose Estimation & Feature Engineering** ([src/preprocessing/preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py))

Extracts 3D pose landmarks from each video frame using MediaPipe's Pose Landmarker model (full, ~6MB). The 33 landmarks are normalized for scale and translation invariance (pelvis-centered, torso-length scaled) and converted into biomechanical features: 13 joint angles (elbow, shoulder, hip, knee, ankle, wrist, torso lean), 6 distance features (ear-shoulder, wrist-shoulder, elbow-hip), and 18 view-specific specialized features targeting exercise-specific confusion patterns. The temporal sequence is resampled to T=50 frames via linear interpolation, yielding a (50, 37) feature tensor per repetition.

**Stage 2 — Exercise Recognition** ([src/scripts/experiment_1.py](src/scripts/experiment_1.py), [src/models/model_builder.py](src/models/model_builder.py))

Flattens temporal features into a 1850-dimensional vector (50 × 37) and passes through a 3-layer feedforward network [512→256→128] with ReLU + Dropout to classify the exercise into one of 15 classes. The model is trained subject-disjoint (train/val/test sets contain completely different volunteers) with 30 randomized runs for statistical robustness. Achieves 90.36% macro F1 (side) and 86.94% macro F1 (front).

**Stage 3 — Per-Repetition Assessment** ([src/scripts/video_to_assessment_cnn_all.py](src/scripts/video_to_assessment_cnn_all.py), [src/models/assessment_models/](src/models/assessment_models/))

Segments individual repetitions from the pose stream using exercise/view-specific biomechanical signals (e.g., arm elevation for raises, elbow flexion for curls, hip/knee depth for squats), then passes each repetition through a compact temporal CNN regressor. Generates 5 aspect-level quality scores (0–10 scale) per repetition, enabling downstream analysis of fatigue, consistency, and improvement over the set. Returns variable N_reps per video.

**Stage 4 — Coaching Agent** ([src/agents/coaching_agent.py](src/agents/coaching_agent.py))

A LangGraph state machine with 5 sequential nodes: (1) load exercise-specific assessment criteria; (2) analyze aggregated scores to generate system warnings; (3) compute rule-based trend analysis (early vs. late reps, fatigue detection, consistency score); (4) invoke Gemini 2.5 Flash LLM to generate natural language feedback grounded in per-rep data; (5) format final response as structured JSON. All deterministic analysis happens in Python (Node 3) to reduce token usage and latency; the LLM only receives pre-computed summaries.

---

## Dataset

### Participants

- **51 volunteers**: 41 male (80.4%), 10 female (19.6%)
- **Age range**: 15–29 years (mean 21.3 ± 3.1)
- **BMI range**: 18–37 kg/m² (mean 24.7 ± 4.1)
- **All participants**: recreationally active, no musculoskeletal injuries at recording time

| Attribute | Min | Max | Mean ± SD |
|-----------|-----|-----|-----------|
| Age (years) | 15 | 29 | 21.3 ± 3.1 |
| Height (cm) | 158 | 190 | 174.2 ± 7.6 |
| Weight (kg) | 52 | 110 | 77.5 ± 13.9 |
| BMI (kg/m²) | 18 | 37 | 24.7 ± 4.1 |

### Exercise Set

**15 resistance-training exercises** covering upper and lower body:

| ID | Exercise | ID | Exercise |
|----|----------|----|-|
| 1 | Dumbbell Shoulder Press | 9 | Shrugs |
| 2 | Hammer Curls | 10 | Weighted Squats |
| 3 | Standing Dumbbell Front Raises | 11 | Seated Biceps Curls |
| 4 | Lateral Raises | 12 | Triceps Kickbacks |
| 5 | Bulgarian Split Squat | 13 | Rows |
| 6 | EZ Bar Curls | 14 | Deadlift |
| 7 | Incline Dumbbell Bench Press | 15 | Calf Raises |
| 8 | Overhead Triceps Extension | — | — |

### Recording Protocol

- **Camera**: Smartphone (1080×1920 resolution, 30 fps)
- **Views**: Front and side simultaneously (154 videos per view = 308 total)
- **Duration**: 30–60 seconds per set, 8–12 reps per video
- **Setup**: Indoors at multiple local gym facilities; mixed lighting conditions
- **Weight**: Self-selected to reflect realistic training conditions
- **Result**: Substantial inter- and intra-subject variability in execution quality

### Annotation

- **Raters**: 2 certified fitness coaches
- **Level**: Set-level annotation (one label per video)
- **Criteria**: 5 biomechanical aspects per exercise (joint alignment, range of motion, stability, control, tempo)
- **Weighting**: $y = 0.25 \cdot y_{C1} + 0.75 \cdot y_{C2}$ (reliability-weighted fusion)
- **Scale**: 0–10 per criterion

### Dataset Limitations

- **Gender imbalance**: 80.4% male, reflecting cultural constraints in data collection
- **Limited lower-body samples**: More upper-body exercises in the dataset
- **Dataset size**: 51 subjects is modest; performance varies across random splits

---

## Project Structure

<details>
<summary><b>Full Directory Tree</b></summary>

```
ai-virtual-coach/
├── config/                                    # Experiment YAML configs
│   ├── experiment_1_baseline_front.yaml
│   ├── experiment_1_baseline_side.yaml
│   ├── experiment_1_specialized_front.yaml
│   ├── experiment_1_specialized_side.yaml
│   └── archive/                               # Legacy GEI configs
│
├── datasets/
│   ├── Clips/                                 # Raw exercise video clips (15 exercises)
│   ├── GEIs_of_rgb_front/                    # Front-view GEI images (legacy)
│   ├── GEIs_of_rgb_side/                     # Side-view GEI images (legacy)
│   ├── Mediapipe pose estimates/              # ★ Active NPZ pose data
│   │   ├── pose_data_front_19_features.npz
│   │   └── pose_data_side_19_features.npz
│   ├── old pose estimates/                    # Legacy NPZ files
│   ├── ppl in background/
│   ├── Annotation aspects.txt                 # Assessment criteria
│   ├── metadata.xlsx.csv                      # Subject demographics
│   ├── pose_landmarker_full.task              # MediaPipe model (full)
│   └── pose_landmarker_lite.task              # MediaPipe model (lite)
│
├── src/                                       # ★ Source code
│   ├── agents/                                # Coaching feedback agent
│   │   ├── __init__.py
│   │   ├── coaching_agent.py                  # LangGraph workflow
│   │   ├── config.py                          # Agent config & API keys
│   │   ├── exercise_criteria.py               # 15 exercises × 5 criteria
│   │   ├── prompts.py                         # LLM prompt templates
│   │   └── state.py                           # Pydantic state models
│   │
│   ├── data/                                  # Data loading & preprocessing
│   │   ├── data_loader.py                     # load_pose_data, split_by_subject
│   │   ├── preprocessing.py                   # prep_tensors, normalize
│   │   ├── augmentation.py                    # Data augmentation pipelines
│   │   └── dataset_builder.py                 # tf.data.Dataset builders
│   │
│   ├── models/                                # Model architectures
│   │   ├── model_builder.py                   # build_model_*, BACKBONE_REGISTRY
│   │   └── assessment_models/                 # 30 serialized .joblib files
│   │       ├── 1__Dumbbell_shoulder_press_best_model.joblib
│   │       ├── 1__Dumbbell_shoulder_press_scaler.joblib
│   │       ├── 2__Hummer_curls_best_model.joblib
│   │       ├── 2__Hummer_curls_scaler.joblib
│   │       └── ... (15 exercises × 2 files each)
│   │
│   ├── scripts/                               # Training & inference scripts
│   │   ├── experiment_1.py                    # ★ Pose MLP training
│   │   ├── vc_core.py                         # Video → assessment utilities
│   │   ├── video_to_assessment_cnn_all.py     # End-to-end video assessment
│   │   ├── gei_embeddings.py                  # GEI embeddings (legacy)
│   │   ├── gei_grid_search_utils.py           # GEI grid-search (legacy)
│   │   └── archive/                           # Deprecated GEI scripts
│   │
│   ├── pipelines/
│   │   └── backend_api_contract.md            # Flutter ↔ Backend API spec
│   │
│   ├── preprocessing/                         # Raw data preprocessing
│   │   ├── preprocess_pose_RGB.py             # MediaPipe pose extraction
│   │   └── analyze_frame_distribution.py      # Frame/FPS analysis
│   │
│   └── utils/                                 # Shared utilities
│       ├── io_utils.py                        # Folder mgmt, seed setting
│       ├── metrics.py                         # Tracking, metrics, param counting
│       └── visualization.py                   # Plots, confusion matrices
│
├── notebooks/
│   ├── EDA/
│   │   └── Analysis.ipynb                     # Exploratory data analysis
│   ├── exer_recog/
│   │   ├── 01_pose_mlp.ipynb                  # ★ Main experiment notebook
│   │   ├── 99_comparison.ipynb                # Cross-experiment comparison
│   │   └── archive/                           # Archived GEI notebooks
│   └── pose_preprocessing/
│       ├── 00_pose_preprocessing.ipynb        # Pose extraction workflow
│       └── 00b_frame_distribution_analysis.ipynb
│
├── output/
│   ├── exer_recog/
│   │   ├── exp_01_pose_mlp_baseline/          # 19-feature results
│   │   ├── exp_01_pose_mlp_specialized/       # 37-feature results
│   │   ├── exp_01_baseline/ … exp_05_small_cnn/ # Legacy GEI results
│   └── assessment/
│       ├── outputs_front/                     # Assessment results (front)
│       └── outputs_side/                      # Assessment results (side)
│
├── tests/
│   ├── __init__.py
│   ├── test_experiment_1.py                   # MLP validation tests
│   └── test_05_small_cnn.py                   # Legacy GEI test
│
├── docs/
│   ├── PROJECT_STRUCTURE.md                   # ⚠️ See README (consolidated)
│   ├── FEATURE_ENGINEERING.md                 # ⚠️ See README §8.1
│   ├── SUBJECT_WISE_SPLITTING_METHODOLOGY.md  # ⚠️ See README §8.3
│   ├── COACHING_AGENT_DOCUMENTATION.md        # ⚠️ See README §8.5
│   ├── Research_Paper_Draft/
│   │   └── main.tex                           # Research paper
│   ├── Final_Thesis/
│   │   └── main.tex                           # Bachelor's thesis
│   └── archive/                               # Historical docs
│
├── plots/
│   ├── frame_distribution_analysis/
│   └── ...
│
├── .env.example                               # Environment template
├── .env                                       # ⚠️ DO NOT COMMIT (git-ignored)
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # ⚠️ TBD (see below)
└── README.md                                  # ← This file
```

</details>

### Top-Level Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| **config/** | YAML experiment configurations (4 active: baseline + specialized × front + side) |
| **datasets/** | Raw video clips, pose data (NPZ), GEI images, annotations, MediaPipe models |
| **src/** | Core source code: agents, data pipeline, models, scripts, utilities |
| **notebooks/** | Interactive Jupyter notebooks for EDA, preprocessing, and main experiment |
| **output/** | Training results, checkpoints, metrics (organized by experiment) |
| **tests/** | Pytest test suite for validation |
| **docs/** | Technical documentation, research paper, thesis |
| **plots/** | Static plots and figures |

---

## Installation & Setup

### Prerequisites

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows (with WSL2 recommended)
- **GPU** (optional): CUDA-capable GPU for faster training; CPU works but slower
- **Disk space**: ~30 GB for datasets + checkpoints

### Step 1: Clone the Repository

```bash
git clone https://github.com/AhmedAly0/AI-Virtual-Coach.git
cd ai-virtual-coach
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n coach python=3.10
conda activate coach
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `tensorflow>=2.13.0` — Exercise recognition models
- `torch>=2.0.0` — Assessment models (temporal CNN)
- `mediapipe>=0.10.0` — Pose extraction
- `langgraph`, `langchain`, `google-generativeai` — Coaching agent
- `scikit-learn`, `joblib` — Assessment inference
- `pydantic>=2.0.0` — Data validation

### Step 4: Download MediaPipe Models

The MediaPipe Pose Landmarker models are already in `datasets/`:
- `pose_landmarker_full.task` (~6 MB) — Full model, high accuracy
- `pose_landmarker_lite.task` (~4 MB) — Lite model, faster

No additional downloads needed.

### Step 5: Configure Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Gemini API key
# Get one free at: https://aistudio.google.com/apikey
nano .env
```

**`.env` template:**
```dotenv
# Gemini API Key (required for coaching agent)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override default model
# GEMINI_MODEL_NAME=gemini-2.5-flash
```

### Step 6: Verify Installation

```bash
# Run a quick validation test
pytest tests/test_experiment_1.py -v

# Or test the coaching agent
python -c "from src.agents import CoachingAgent; print('✓ Coaching agent imported successfully')"
```

---

## Quick Start

### Usage Pattern A: Exercise Recognition Training

Train the exercise recognition model (MLP) from pose data.

```bash
# Single run (baseline, front view)
python src/scripts/experiment_1.py \
  --npz_path datasets/Mediapipe\ pose\ estimates/pose_data_front_19_features.npz \
  --config_path config/experiment_1_baseline_front.yaml

# Multi-run (30 seeds, specialized features, side view)
python src/scripts/experiment_1.py \
  --npz_path datasets/Mediapipe\ pose\ estimates/pose_data_side_19_features.npz \
  --config_path config/experiment_1_specialized_side.yaml \
  --multi_run True
```

**Output**: Results saved to `output/exer_recog/exp_01_pose_mlp_baseline/` or `_specialized/`

**Config file example** (`config/experiment_1_specialized_front.yaml`):
```yaml
data:
  npz_path: datasets/Mediapipe pose estimates/pose_data_front_19_features.npz
  feature_type: front_all_extended  # 37 features

model:
  hidden_layers: [512, 256, 128]
  dropout: 0.4
  num_classes: 15

training:
  batch_size: 16
  learning_rate: 0.00005
  max_epochs: 220
  callbacks:
    early_stopping_patience: 65
    reduce_lr_patience: 15

multi_run:
  num_runs: 30
  base_seed: 42

results:
  base_dir: output/exer_recog/exp_01_pose_mlp_specialized/front
```

### Usage Pattern B: End-to-End Video → Assessment

Process a single exercise video through the full pipeline (recognition + assessment + coaching).

```python
from src.scripts.video_to_assessment_cnn_all import process_exercise_video
from src.agents import CoachingAgent

# Step 1: Process video
results = process_exercise_video(
    video_path="path/to/exercise_video.mp4",
    view="front",
    exercise_id=1,  # Dumbbell Shoulder Press
    model_path="src/models/assessment_models/"
)

# Step 2: Generate coaching feedback
agent = CoachingAgent()
feedback = agent.generate_feedback(
    exercise_id=results["exercise_id"],
    exercise_name=results["exercise_name"],
    rep_scores=results["per_rep_scores"],
    recognition_confidence=results["confidence"],
    view_type="front"
)

print(feedback.feedback_summary)
# Output: "Fantastic effort on your Dumbbell Shoulder Press! ..."
```

### Usage Pattern C: Standalone Coaching Agent

Invoke the coaching agent directly with pre-computed assessment scores.

```python
from src.agents import CoachingAgent, PerRepScore

agent = CoachingAgent()

# Pre-computed per-rep scores (from Stage 3 assessment model)
rep_scores = [
    {
        "rep_number": 1,
        "scores": {
            "Starting position": 9.0,
            "Top position": 8.5,
            "Elbow path": 8.0,
            "Tempo": 9.0,
            "Core stability": 8.5
        }
    },
    {
        "rep_number": 2,
        "scores": {
            "Starting position": 8.8,
            "Top position": 8.3,
            "Elbow path": 8.2,
            "Tempo": 8.8,
            "Core stability": 8.5
        }
    },
    # ... 10 more reps with declining scores
]

response = agent.generate_feedback(
    exercise_id=1,
    exercise_name="Dumbbell Shoulder Press",
    rep_scores=rep_scores,
    recognition_confidence=0.92,
    view_type="front"
)

print(f"Overall score: {response.overall_score}")
print(f"Fatigue detected: {response.fatigue_detected}")
print(f"Feedback: {response.feedback_summary}")
```

**Expected output:**
```
Overall score: 7.4
Fatigue detected: True
Feedback: Fantastic effort on your Dumbbell Shoulder Press! Your starting
position was consistently strong throughout the set. However, I noticed your
elbow path became less vertical as the set progressed, particularly from rep 9
onward. It looks like fatigue started to set in towards the end — your top
position and core stability both dropped noticeably in the final reps...
```

### Usage Pattern D: Backend API Endpoint

The backend API contract is documented at [src/pipelines/backend_api_contract.md](src/pipelines/backend_api_contract.md).

**Endpoint**: `POST /api/session/analyze`

**Request**:
```json
{
  "exercise_view": "front",
  "pose_sequence": [
    [[0.5, 0.5, 0.0, 0.99], [0.5, 0.5, 0.0, 0.99], ...],  // Frame 1 (33 landmarks × 4 values)
    [[0.5, 0.5, 0.0, 0.99], [0.5, 0.5, 0.0, 0.99], ...],  // Frame 2
    ...
  ],
  "metadata": {
    "fps": 24.5,
    "frame_count": 734,
    "device": "mobile"
  }
}
```

**Response (200 OK)**:
```json
{
  "exercise": "Dumbbell Shoulder Press",
  "reps_detected": 12,
  "scores": {
    "Starting position": 7.9,
    "Top position": 7.0,
    "Elbow path": 6.8,
    "Tempo": 7.7,
    "Core stability": 7.4
  },
  "overall_score": 7.4,
  "feedback": [
    "Fantastic effort!",
    "Focus on keeping elbows vertical.",
    "Form degradation in final reps — consider more rest."
  ]
}
```

---

## Technical Deep Dives

<details>
<summary><b>§8.1 Pose Estimation & Feature Engineering</b></summary>

### Pose Extraction (MediaPipe)

Each video frame is processed by **MediaPipe Pose Landmarker (Full)** to extract 33 3D body landmarks:

```
Landmark indices:
├── Head/Face: 0 (nose), 1-10 (eyes, ears, mouth)
├── Torso: 11-12 (shoulders), 23-24 (hips)
├── Arms: 13-16 (elbows, wrists), 17-22 (pinky, index, thumb)
└── Legs: 25-32 (knees, ankles, heels, foot index)
```

Each landmark has 4 values: `[x, y, z, visibility]`
- **x, y**: Normalized image-plane coordinates [0, 1]
- **z**: Depth relative to hip midpoint (negative = closer to camera)
- **visibility**: Confidence score [0, 1]

### 3D Normalization

All landmarks are normalized for **scale and translation invariance** using pelvis-centered, torso-length-scaled transformation:

$$\text{pelvis} = \frac{1}{2}(h_{23} + h_{24})$$

$$\text{mid\_shoulder} = \frac{1}{2}(h_{11} + h_{12})$$

$$L = \|\text{mid\_shoulder} - \text{pelvis}\|_2 = \sqrt{(s_x - p_x)^2 + (s_y - p_y)^2 + (s_z - p_z)^2}$$

$$h_i^{\text{norm}} = \frac{h_i - \text{pelvis}}{L}$$

**Benefits**: Camera distance, body size, and subject differences become irrelevant — features depend only on movement quality.

### Base Feature Set (19 Features)

#### Joint Angles (13)

| Feature | Description | Angle Formula |
|---------|-------------|---------------|
| Left/Right Elbow | Arm flexion | shoulder→elbow→wrist |
| Left/Right Shoulder | Arm abduction | elbow→shoulder→hip |
| Left/Right Hip | Hip flexion | shoulder→hip→knee |
| Left/Right Knee | Knee flexion | hip→knee→ankle |
| Torso Lean | Forward lean from vertical | pelvis-shoulder axis vs gravity |
| Left/Right Ankle | Ankle dorsiflexion | knee→ankle→heel |
| Left/Right Wrist | Wrist angle | elbow→wrist→pinky |

Computed using 3D dot product:
$$\theta = \arccos\left(\frac{\vec{v_1} \cdot \vec{v_2}}{\|\vec{v_1}\| \|\vec{v_2}\|}\right)$$

#### Distance Features (6)

| Feature | Description | Use Case |
|---------|-------------|----------|
| Left/Right Ear-Shoulder (vertical) | Vertical gap | Shrug detection |
| Left/Right Wrist-Shoulder (Euclidean) | Arm extension distance | Pressing, raising |
| Left/Right Elbow-Hip (Euclidean) | Elbow proximity to torso | Arm tuck position |

### View-Specific Specialized Features (18 each)

Different camera angles capture different biomechanics. We engineered view-specific features to address confusion clusters.

#### Front View Specialized Features (18)

**Cluster 1 — Curl Variants (Hammer ↔ EZ Bar ↔ Seated):**
- `forearm_supination` — Grip angle (neutral ~0°, angled ~30-45°, full ~90°)
- `upper_arm_vertical` — Arm position (inclined bench vs standing)
- `inter_wrist_distance` — Bar width (fixed in EZ Bar)
- `elbow_body_distance` — Elbow flare (tucked vs flared)

**Cluster 2 — Hinge Movements (Deadlift ↔ Rows):**
- `shoulder_width_ratio` — Increases with scapular retraction (rows)
- `wrist_hip_vertical` — Bar rises (deadlift) vs arm pulls (rows)
- `hip_depth_ratio` — Hip position change (deadlift) vs fixed (rows)

**Cluster 3 — Arm Extensions (Kickbacks ↔ Rows):**
- `wrist_posterior_z` — Wrist behind hip (kickbacks) vs beside hip (rows)

**Cluster 4 — Minimal Motion (Shrugs ↔ Calf Raises):**
- `heel_elevation` — Heel rise (calf raises)
- `shoulder_center_y` — Shoulder elevation (shrugs)
- `shoulder_hip_y_ratio`, `ankle_center_y` — Context

#### Side View Specialized Features (18)

**Group 1 — Vertical Displacement (4 features):** Shrugs vs Calf Raises
- `shoulder_elevation_y`, `heel_ground_clearance`, `shoulder_hip_y_ratio`, `ear_shoulder_compression`

**Group 2 — Overhead Arm Position (4):** Overhead Triceps Extension discrimination
- `elbow_above_shoulder`, `wrist_above_elbow`, `upper_arm_vertical_angle_side`, `forearm_vertical_angle_side`

**Group 3 — Sagittal Arm Trajectory (4):** Front-to-back movement patterns
- `wrist_forward_of_shoulder`, `elbow_forward_of_hip`, `arm_reach_forward`, `elbow_tuck_side`

**Group 4 — Hip Hinge Profile (4):** Deadlift vs Rows vs Kickbacks
- `torso_angle_from_vertical`, `hip_behind_ankle`, `shoulder_forward_of_hip`, `knee_hip_alignment_z`

**Group 5 — Postural Stability (2):** General context
- `stance_width_normalized`, `center_of_mass_y`

### Temporal Processing

**Fixed-length resampling** to T=50 frames via linear interpolation:
```
Original video: T_orig = 112 frames
Resampled:      T_fixed = 50 frames
                → (50, 37) feature tensor per repetition
```

**Tempo preservation** — Since resampling destroys speed information, preserve separately:
- `tempo_duration_sec` — Video duration in seconds
- `tempo_frame_count` — Original frame count
- `tempo_fps` — Video FPS

### Final Feature Vectors

**Front View (37 features)**:
- Base: 19 (13 angles + 6 distances)
- Specialized: 18 (curl, hinge, kickback, elevation discrimination)
- **Total: 37**

**Side View (37 features)**:
- Base: 19 (13 angles + 6 distances)
- Specialized: 18 (vertical displacement, overhead, sagittal, hip hinge, stability)
- **Total: 37**

### Implementation Reference

- Extraction: [src/preprocessing/preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py)
- Data loading: [src/data/data_loader.py](src/data/data_loader.py)
- Preprocessing notebook: [notebooks/pose_preprocessing/](notebooks/pose_preprocessing/)

</details>

<details>
<summary><b>§8.2 Exercise Recognition Module</b></summary>

### Architecture

**Input**: Flattened temporal pose features (50 × 37 = 1850 dimensions)  
**Hidden layers**: [512, 256, 128] with ReLU + BatchNorm + Dropout(0.35-0.4)  
**Output**: 15-class softmax (exercise probabilities)

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(1850,)),
    BatchNormalization(),
    Dropout(0.35),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),
    Dense(15, activation='softmax')
])
```

### Training Protocol

- **Optimizer**: Adam with learning rate 5e-5 to 6e-5
- **Batch size**: 16
- **Epochs**: 200–220
- **Early stopping**: Patience 60–65 epochs (monitoring validation loss)
- **Learning rate reduction**: 0.5× factor, patience 15 epochs
- **Data split**: Subject-disjoint 3-way (55% train, 15% val, 30% test)
- **Stratification**: All 15 exercises in all splits
- **Multi-run**: 30 independent runs with different random seeds

### Results

| View | Accuracy | Macro F1 | Details |
|------|----------|----------|---------|
| **Side** | 90.49 ± 2.93% | 90.36 ± 2.92% | Clearer sagittal movements, better elbow/torso visibility |
| **Front** | 87.13 ± 3.42% | 86.94 ± 3.55% | Bilateral symmetry captured, but depth-axis less reliable |

**Key findings**:
- Side view outperforms front by ~3.4% due to superior sagittal-plane visibility
- Specialized features crucial: baseline (19 features) → ~75% accuracy; specialized (37 features) → ~90% (side), ~87% (front)
- Subject-disjoint protocol ensures generalization; test subjects never seen during training

**Per-class performance (side view)**:
- Best: Deadlift (0.97), Bulgarian Split Squat (0.96), Seated Biceps Curls (0.96), Rows (0.95)
- Worst: Hammer Curls (0.63) — residual confusion with other curl variants
- Strong overhead exercises: Overhead Triceps Extension (0.92)

### Configuration

Config files in `config/`:
- `experiment_1_baseline_front.yaml` — 19 features, front view
- `experiment_1_baseline_side.yaml` — 19 features, side view
- `experiment_1_specialized_front.yaml` — 37 features, front view
- `experiment_1_specialized_side.yaml` — 37 features, side view

### Usage

```python
from src.scripts.experiment_1 import train_experiment_1, train_experiment_1_multi_run

# Single run
results = train_experiment_1(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_specialized_front.yaml'
)

# Multi-run (30 seeds)
all_runs, stats = train_experiment_1_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_specialized_front.yaml'
)
print(f"Mean F1: {stats['macro_f1_mean']:.2%} ± {stats['macro_f1_std']:.2%}")
```

### Source Files

- Training script: [src/scripts/experiment_1.py](src/scripts/experiment_1.py)
- Model builder: [src/models/model_builder.py](src/models/model_builder.py)
- Notebook: [notebooks/exer_recog/01_pose_mlp.ipynb](notebooks/exer_recog/01_pose_mlp.ipynb)

</details>

<details>
<summary><b>§8.3 Subject-Wise Splitting Methodology</b></summary>

### Why Subject-Wise Splitting?

**Problem with sample-level splitting**:
- Same subject appears in train AND test sets
- Model learns subject body shape, not exercise movements
- Test accuracy inflated; real-world generalization fails

**Solution**:
```
IF subject_id = "volunteer_007" is in training set
THEN ALL videos of volunteer_007 are in training set
     AND volunteer_007 has ZERO videos in test set
```

**Benefits**: Tests true generalization; prevents data leakage; realistic deployment metrics.

### Three Splitting Patterns

#### Pattern 1: Two-Way Split (Train/Test)

```python
from src.data.data_loader import split_by_subject_two_way

train_samples, test_samples = split_by_subject_two_way(
    dataset=all_samples,
    split_ratio=0.3,        # 30% subjects in test
    seed=42,
    stratified=True         # All 15 exercises in both splits
)
# Result: ~70% train subjects, ~30% test subjects
```

#### Pattern 2: Three-Way Split (Train/Val/Test)

```python
from src.data.data_loader import split_by_subjects_three_way

train, val, test = split_by_subjects_three_way(
    dataset=all_samples,
    val_ratio=0.15,         # 15% subjects for validation
    test_ratio=0.3,         # 30% for test
    seed=42,
    stratified=True         # All classes in all splits
)
# Result: ~55% train, ~15% val, ~30% test
```

#### Pattern 3: K-Fold Cross-Validation

```python
from src.data.data_loader import build_subject_folds

folds = build_subject_folds(
    dataset=pool_samples,
    num_folds=5,
    seed=42,
    stratified=True
)

for fold_idx, val_fold in enumerate(folds):
    train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
    train_samples = sum(train_folds, [])
    val_samples = val_fold
    # Train and evaluate
```

### Stratification Algorithm

Ensures **all 15 exercise classes appear in all splits** (critical for rare exercises):

```
FOR EACH exercise class:
    IF exercise has 1 subject   → Assign to train only (warn)
    IF exercise has 2 subjects  → 1 to train, 1 to test
    IF exercise has 10+ subjects → ~70% train, ~30% test
    
Verify: No overlap between splits (zero data leakage)
```

### Verification Utility

```python
from src.data.data_loader import verify_subject_split_integrity

results = verify_subject_split_integrity(train, val, test, verbose=True)

assert not results['has_subject_overlap'],    "Data leakage!"
assert results['test_classes'] == 15,         "Classes missing from test!"
```

### Subject ID Normalization

Folder names vary; canonicalize to `volunteer_NNN`:

```
"Volunteer #1"     → "volunteer_001"
"volunteer #10"    → "volunteer_010"
"v20"              → "volunteer_020"
"Volunteer_31"     → "volunteer_031"
```

### Best Practices

1. **Always use explicit seeds** for reproducibility
2. **Enable stratification** for final experiments
3. **Verify splits** before training (`verify_subject_split_integrity`)
4. **Never augment test/val data** (only training)
5. **Use stratified splits** even for small datasets

### Source Files

- Core logic: [src/data/data_loader.py](src/data/data_loader.py)
- Convenience wrappers: [src/data/dataset_builder.py](src/data/dataset_builder.py)

</details>

<details>
<summary><b>§8.4 Exercise Assessment Module</b></summary>

### Overview

The assessment module converts recognized exercise videos into **per-rep quality scores** across 5 biomechanical aspects. Unlike simple "overall score" systems, it provides granular feedback enabling fatigue detection and aspect-specific coaching.

### Repetition Segmentation

Each video is segmented into individual reps using an exercise/view-specific **1D biomechanical signal**:
- **Press/Raise**: Arm elevation (Y-position of wrist)
- **Curl**: Elbow flexion angle
- **Squat**: Knee depth (hip-knee angle)
- **Deadlift**: Hip hinge depth

Signal is smoothed and rep boundaries detected via **threshold-based cycle logic** (up→down→up or down→up→down) with:
- Adaptive thresholds from within-video statistics
- Hysteresis to prevent noise-induced flipping
- Minimum duration and separation constraints

**Output**: Variable N_reps per video (typically 8–12)

### Temporal CNN Architecture

```
Input: (50, 9) temporal joint angle sequence per rep
  ↓
Conv1d(9→64, kernel=3, stride=1, padding='same')
ReLU + Dropout(0.3)
  ↓
Conv1d(64→128, kernel=3, stride=1, padding='same')
ReLU + Dropout(0.3)
  ↓
AdaptiveAvgPool1d(output_size=1)
  → (128,) embedding per rep
  ↓
Attention pooling across N_reps
  → (128,) subject embedding
  ↓
Dense(5) → 5 aspect scores (0–10 scale per rep)
```

### 30 Pre-trained Models

15 exercises × 2 views = 30 `.joblib` model files:

```
src/models/assessment_models/
├── 1__Dumbbell_shoulder_press_best_model.joblib
├── 1__Dumbbell_shoulder_press_scaler.joblib
├── 2__Hummer_curls_best_model.joblib
├── 2__Hummer_curls_scaler.joblib
├── ...
└── 15__Calf_raises_scaler.joblib
```

Each model is trained separately per exercise/view under subject-disjoint splits. Scalers normalize features before inference.

### Assessment Accuracy

**Mean Absolute Error (0–10 scale)**, reported mean ± std over 10 runs:

| Exercise | Front | Side | Notes |
|----------|-------|------|-------|
| Rows | 2.26 ± 0.86 | 2.14 ± 0.99 | Best performance |
| Standing Front Raises | 2.83 ± 1.01 | 2.64 ± 1.17 | Clear arm trajectory |
| Dumbbell Shoulder Press | 3.43 ± 0.75 | 3.49 ± 0.60 | Reproducible start position |
| Triceps Kickbacks | 4.80 ± 0.92 | 4.61 ± 0.65 | Most challenging |
| Deadlift | 4.53 ± 0.98 | 4.68 ± 1.36 | Complex movement |
| **Overall** | **3.54 ± 1.21** | **3.51 ± 1.26** | ~35 points on 0–100 scale |

### 5 Assessment Criteria per Exercise

All 15 exercises have 5 biomechanical criteria (from [src/agents/exercise_criteria.py](src/agents/exercise_criteria.py)):

| Exercise | Criteria |
|----------|----------|
| **Dumbbell Shoulder Press** | Starting position, Top position, Elbow path, Tempo, Core stability |
| **Hammer Curls** | Elbow position, Wrist orientation, Movement control, Range of motion, Tempo |
| **Standing Front Raises** | Arm path, Raise height, Core engagement, Wrist alignment, Controlled tempo |
| **Lateral Raises** | Arm angle, Raise height, Elbow position, Trap engagement, No momentum |
| **Bulgarian Split Squat** | Rear foot, Front knee tracking, Torso angle, Depth, Balance |
| **EZ Bar Curls** | Grip, Elbow position, Bar path, Full ROM, No shoulder swing |
| **Incline DB Bench Press** | Bench angle, Dumbbell path, Elbow angle, Back position, Wrist stability |
| **Overhead Triceps Extension** | Elbow position, ROM, Wrist position, Core tightness, Tempo |
| **Shrugs** | Shoulder path, ROM, Neck stability, Weight control, Pause at top |
| **Weighted Squats** | Stance, Depth, Knee tracking, Back posture, Weight distribution |
| **Seated Biceps Curls** | Elbow position, Shoulder stability, ROM, Wrist position, Back support |
| **Triceps Kickbacks** | Upper arm position, Elbow movement, Full extension, No torso swing, Neutral spine |
| **Rows** | Back angle, Elbow path, Shoulder retraction, Pulling motion, Stable base |
| **Deadlift** | Spine, Hips, Bar path, Lockout, Tempo |
| **Calf Raises** | ROM, Knee status, Balance, Tempo, Peak hold |

### Subject-Level Aggregation

Since labels are **set-level** (one per video) but predictions are **rep-level**, we aggregate:

$$e_r = \text{TemporalCNN}(rep_r) \quad (\text{128-dim embedding per rep})$$

$$\alpha_r = \frac{\exp(a(e_r))}{\sum_k \exp(a(e_k))} \quad (\text{attention weights})$$

$$e_{\text{subject}} = \sum_r \alpha_r e_r \quad (\text{subject embedding})$$

$$\hat{y}_{\text{aspect}} = \text{Dense}(e_{\text{subject}}) \quad (\text{5 aspect scores})$$

This enables learning from set-level labels while providing rep-level insights.

### Training Objective

- **Loss**: Mean Squared Error on normalized [0, 1] scale
- **Targets**: Reliability-weighted coach annotations: $y = 0.25 \cdot y_{C1} + 0.75 \cdot y_{C2}$
- **Inference**: Rescale predictions to [0, 10] scale

### Usage

```python
from src.scripts.video_to_assessment_cnn_all import process_exercise_video

results = process_exercise_video(
    video_path="exercise.mp4",
    view="front",
    exercise_id=1,
    model_path="src/models/assessment_models/"
)

print(f"Exercise: {results['exercise_name']}")
print(f"Reps detected: {results['n_reps']}")
print(f"Per-rep scores:")
for rep in results['per_rep_scores']:
    print(f"  Rep {rep['rep_number']}: {rep['scores']}")
```

### Source Files

- End-to-end script: [src/scripts/video_to_assessment_cnn_all.py](src/scripts/video_to_assessment_cnn_all.py)
- Rep segmentation: [src/scripts/vc_core.py](src/scripts/vc_core.py)
- Pre-trained models: [src/models/assessment_models/](src/models/assessment_models/)

</details>

<details>
<summary><b>§8.5 Coaching Agent (LangGraph + Gemini)</b></summary>

### Architecture

The coaching agent is a **5-node LangGraph state machine** that transforms per-rep scores into natural language feedback:

```
START
  ├─► Node 1: load_criteria
  │     Load 5 biomechanical criteria for the exercise
  │
  ├─► Node 2: analyze_scores
  │     Generate warnings if scores are low or confidence is weak
  │
  ├─► Node 3: analyze_rep_trends (DETERMINISTIC)
  │     Compute trend analysis:
  │     - Early vs late rep comparison (first 3 vs last 3)
  │     - Fatigue detection (≥2 criteria declining)
  │     - Consistency score (inverse of variance)
  │     - Strongest/weakest criteria
  │
  ├─► Node 4: generate_llm
  │     Invoke Gemini 2.5 Flash with formatted context
  │     Return natural language feedback
  │
  └─► Node 5: format_response
       Assemble FeedbackResponse with all scores, trends, feedback
  └─► END
```

### Trend Detection Algorithm

For each criterion, compute **early vs late rep performance**:

$$\text{early\_mean} = \text{mean}(\text{first 3 reps})$$
$$\text{late\_mean} = \text{mean}(\text{last 3 reps})$$
$$\text{trend\_diff} = \text{late\_mean} - \text{early\_mean}$$

$$\text{trend} = \begin{cases}
\text{improving} & \text{if trend\_diff} > 0.5 \\
\text{declining} & \text{if trend\_diff} < -0.5 \\
\text{stable} & \text{otherwise}
\end{cases}$$

**Fatigue Detection**:
$$\text{fatigue\_detected} = |\{c : \text{trend}(c) = \text{declining}\}| \geq 2$$

Threshold of 2 reduces false positives; genuine fatigue affects multiple aspects simultaneously.

### Consistency Score

$$\text{consistency} = \max\left(0, \min\left(10, 10 - 3\bar{\sigma}\right)\right)$$

where $\bar{\sigma}$ is the mean standard deviation across all criteria. Perfect consistency ($\sigma = 0$) → score 10; high variance ($\bar{\sigma} \geq 3.33$) → score 0.

### Data Models (Pydantic)

```python
class PerRepScore:
    rep_number: int          # 1-indexed
    scores: dict[str, float] # criterion → score [0, 10]

class CriterionTrend:
    criterion: str
    mean: float
    std: float
    trend: str              # "improving" / "declining" / "stable"
    trend_magnitude: float
    weakest_reps: list[int]

class RepTrendAnalysis:
    rep_count: int
    criterion_trends: list[CriterionTrend]
    fatigue_detected: bool
    consistency_score: float
    strongest_criterion: str
    weakest_criterion: str
    per_rep_averages: list[float]

class FeedbackResponse:
    exercise_name: str
    exercise_id: int
    recognition_confidence: float
    rep_count: int
    aggregated_scores: dict[str, float]
    overall_score: float
    consistency_score: float
    fatigue_detected: bool
    trends: dict[str, str]
    feedback_summary: str        # LLM-generated
    warnings: list[str]
```

### LLM Prompt Structure

The system prompt establishes the **persona** (expert AI fitness coach), and the human message provides **structured context**:

```
[System]
You are an expert AI Fitness Coach with deep knowledge of exercise
biomechanics, proper form, and injury prevention. Always be encouraging,
provide specific actionable advice, and reference per-rep data...

[Human]
The user performed: {exercise_name} ({rep_count} reps)

Exercise-Specific Assessment Criteria:
{exercise_criteria}  ← Bullet list of 5 criteria

═══ PER-REP ASSESSMENT BREAKDOWN ═══
{per_rep_breakdown}  ← ASCII table (N rows × 6 cols: rep, 5 criteria, avg)

═══ AGGREGATED ANALYSIS ═══
Overall Score: {overall_score}/10
Consistency: {consistency_score}/10
Per-Criterion Summary: {criterion_summary}  ← Trends with ↑↓→ arrows

═══ FATIGUE ANALYSIS ═══
{fatigue_analysis}

═══ YOUR TASK ═══
1. Acknowledge strengths (reference criteria)
2. Identify key issues (reference specific reps)
3. Address fatigue if detected
4. Provide 2–3 actionable tips
5. End with encouragement
Keep under {max_words} words. Conversational tone.
```

### Configuration

All settings in [src/agents/config.py](src/agents/config.py):

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
MAX_FEEDBACK_WORDS = 150
SCORE_THRESHOLDS = {
    "excellent": 8.5,
    "good": 7.0,
    "needs_improvement": 5.0,
    "poor": 0.0
}
```

### Usage Example

```python
from src.agents import CoachingAgent

agent = CoachingAgent()

response = agent.generate_feedback(
    exercise_id=1,
    exercise_name="Dumbbell Shoulder Press",
    rep_scores=[
        {"rep_number": 1, "scores": {"Starting position": 9.0, ...}},
        # ... 11 more reps with declining scores
    ],
    recognition_confidence=0.92,
    view_type="front"
)

print(response.feedback_summary)
# Output: "Fantastic effort on your Dumbbell Shoulder Press! Your
# starting position was consistently strong. However, I noticed your
# elbow path became less vertical as the set progressed, particularly
# from rep 9 onward. Fatigue appeared to set in — your top position and
# core stability both dropped in the final reps. For next time, focus on
# keeping elbows straight up, and consider more rest between sets. Great work!"
```

### Error Handling

All nodes catch exceptions gracefully:

| Failure | Handling |
|---------|----------|
| Unknown exercise ID | `exercise_criteria = []`; LLM gets no criteria context |
| Missing API key | LLM feedback → "API key not configured" |
| API error | LLM feedback → "Error generating feedback: {msg}" |
| Empty rep scores | Analysis proceeds with `overall_score = 0.0`; trends skipped |

### Extensibility

**Adding a new exercise**: Update [src/agents/exercise_criteria.py](src/agents/exercise_criteria.py):
```python
EXERCISE_CRITERIA[16] = (
    "Lunges",
    [
        "Step length: appropriate for hip hinge",
        "Front knee tracking: over toes, not past",
        "Torso: upright posture throughout",
        "Rear knee: controlled descent near floor",
        "Balance: stable with no wobble",
    ]
)
```

**Swapping LLM backend**: Replace `ChatGoogleGenerativeAI` with any LangChain LLM.

### Source Files

- Main orchestrator: [src/agents/coaching_agent.py](src/agents/coaching_agent.py)
- State models: [src/agents/state.py](src/agents/state.py)
- Data models: [src/agents/exercise_criteria.py](src/agents/exercise_criteria.py)
- Prompts: [src/agents/prompts.py](src/agents/prompts.py)
- Config: [src/agents/config.py](src/agents/config.py)

</details>

<details>
<summary><b>§8.6 Backend API Contract</b></summary>

### Endpoint: `POST /api/session/analyze`

Processes a full exercise session's pose data and returns assessment with scores and coaching feedback.

### Request Schema

```json
{
  "exercise_view": "front" | "side",
  "pose_sequence": [
    [[x, y, z, visibility], [x, y, z, visibility], ...],  // 33 landmarks per frame
    ...
  ],
  "metadata": {
    "fps": 24.57,
    "frame_count": 734,
    "device": "mobile"
  }
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `exercise_view` | string | `"front"` or `"side"` | Camera angle |
| `pose_sequence` | float[][][] | frames × 33 × 4 | Pose landmarks (frame-wise) |
| `metadata.fps` | float | > 0 | Actual FPS from device |
| `metadata.frame_count` | int | = len(pose_sequence) | Total frames |
| `metadata.device` | string | `"mobile"` | Always mobile for Phase 1 |

**Landmark format** (4 values per landmark):
- `[0]` = x: horizontal [0, 1]
- `[1]` = y: vertical [0, 1]
- `[2]` = z: depth [-1, 1]
- `[3]` = visibility: confidence [0, 1]

### Response Schema (200 OK)

```json
{
  "exercise": "Dumbbell Shoulder Press",
  "reps_detected": 12,
  "scores": {
    "Starting position": 7.9,
    "Top position": 7.0,
    "Elbow path": 6.8,
    "Tempo": 7.7,
    "Core stability": 7.4
  },
  "overall_score": 7.4,
  "feedback": [
    "Great depth on most reps!",
    "Watch your back angle.",
    "Good knee stability."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `exercise` | string | Detected exercise name |
| `reps_detected` | int | Number of valid reps |
| `scores` | dict[str, float] | Aspect scores, 0–10 scale (frontend multiplies by 10) |
| `overall_score` | float | Grand mean, 0–10 scale |
| `feedback` | list[string] | Coaching tips (bulleted for display) |

### Error Response (4xx/5xx)

```json
{
  "error_code": "NO_REPS_DETECTED",
  "message": "No valid repetitions were detected. Please try again with a clearer view."
}
```

| Status | `error_code` | `message` |
|--------|-------------|----------|
| 400 | `INVALID_REQUEST` | "Invalid request format." |
| 400 | `NO_POSE_DATA` | "No pose data received." |
| 400 | `INSUFFICIENT_FRAMES` | "Too few frames to analyze." |
| 422 | `NO_REPS_DETECTED` | "No valid reps detected." |
| 422 | `UNRECOGNIZED_EXERCISE` | "Could not identify exercise." |
| 500 | `ANALYSIS_FAILED` | "Internal error during analysis." |

### Full Documentation

See [src/pipelines/backend_api_contract.md](src/pipelines/backend_api_contract.md) for complete details, FastAPI skeleton, network setup, and testing examples.

</details>

---

## Configuration Reference

### YAML Experiment Configs

All YAML configs in `config/` follow this structure:

```yaml
data:
  npz_path: datasets/Mediapipe pose estimates/pose_data_{front,side}_19_features.npz
  feature_type: all | front_all_extended | side_all_extended | ...

model:
  hidden_layers: [512, 256, 128]      # MLP hidden layer sizes
  dropout: 0.35-0.4                   # Dropout rate
  num_classes: 15

training:
  batch_size: 16
  learning_rate: 5e-5 to 6e-5
  max_epochs: 200-220

callbacks:
  early_stopping_patience: 60-65
  reduce_lr_patience: 15

multi_run:
  num_runs: 30                        # For multi-run evaluation
  base_seed: 42

results:
  base_dir: output/exer_recog/exp_01_pose_mlp_{baseline,specialized}/{front,side}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Google AI Studio API key for coaching agent |
| `GEMINI_MODEL_NAME` | ❌ | `gemini-2.5-flash` | Gemini model to use (can also use `gemini-2.5-pro`) |

### Score Threshold Definitions

| Category | Threshold | Meaning |
|----------|-----------|---------|
| Excellent | ≥ 8.5 | Expert-level form |
| Good | 7.0–8.4 | Solid form with minor issues |
| Needs Improvement | 5.0–6.9 | Noticeable form breakdown |
| Poor | < 5.0 | Significant form issues |

---

## Experimental Results

### Exercise Recognition Performance

**Setup**: Subject-disjoint 30-run evaluation, 37-feature MLP, stratified splits

| View | Accuracy | Macro F1 | Details |
|------|----------|----------|---------|
| **Side** | 90.49 ± 2.93% | 90.36 ± 2.92% | Best overall; clear sagittal-plane visibility |
| **Front** | 87.13 ± 3.42% | 86.94 ± 3.55% | Good bilateral symmetry; depth less reliable |
| **Improvement** | +3.36% | +3.42% | Side view consistently outperforms |

**Confusion pattern reduction** with specialized features:

| Confusion Cluster | Before (19 feat) | After (37 feat) | Improvement |
|-------------------|------------------|-----------------|-------------|
| Curl Variants | 30–35% | ~5% | -25 pp |
| Hinge Movements | ~25% | ~8% | -17 pp |
| Arm Extensions | ~22% | ~6% | -16 pp |
| Minimal Motion | ~16% | ~4% | -12 pp |

### Exercise Assessment Performance

**Setup**: Subject-disjoint 10-run evaluation, temporal CNN, MAE on 0–10 scale

| View | Overall MAE | Range | Notes |
|------|-------------|-------|-------|
| **Front** | 3.54 ± 1.21 | 2.26–4.80 | ~35 points on 0–100 scale |
| **Side** | 3.51 ± 1.26 | 2.14–4.68 | Comparable to front |

**Best- and worst-performing exercises**:

| Best | MAE | Worst | MAE |
|-----|-----|-------|-----|
| Rows | 2.20 | Triceps Kickbacks | 4.71 |
| Standing Front Raises | 2.74 | Deadlift | 4.61 |
| Seated Biceps Curls | 2.78 | Lateral Raises | 4.42 |

### Key Findings

1. **View Complementarity**: Side view captures elbow/torso dynamics better (sagittal plane); front view captures bilateral symmetry. Both views are valuable.
2. **Feature Engineering Effectiveness**: The systematic approach to specialized features (confusion-driven design) yields significant improvements (3–25 pp accuracy gains).
3. **Generalization**: Subject-disjoint protocol ensures the system generalizes to new users; performance is robust across 30 random seeds.
4. **Assessment Accuracy**: MAE ~3.5/10 is reasonable for practical coaching; enables detection of major form issues and fatigue patterns.

---

## Supported Exercises

All 15 resistance-training exercises with their 5 assessment criteria:

| ID | Exercise | Criteria |
|----|----------|----------|
| **1** | **Dumbbell Shoulder Press** | Starting position • Top position • Elbow path • Tempo • Core stability |
| **2** | **Hammer Curls** | Elbow position • Wrist orientation • Movement control • Range of motion • Tempo |
| **3** | **Standing Dumbbell Front Raises** | Arm path • Raise height • Core engagement • Wrist alignment • Controlled tempo |
| **4** | **Lateral Raises** | Arm angle • Raise height • Elbow position • Trap engagement • No momentum |
| **5** | **Bulgarian Split Squat** | Rear foot • Front knee tracking • Torso angle • Depth • Balance |
| **6** | **EZ Bar Curls** | Grip • Elbow position • Bar path • Full ROM • No shoulder swing |
| **7** | **Incline Dumbbell Bench Press** | Bench angle • Dumbbell path • Elbow angle • Back position • Wrist stability |
| **8** | **Overhead Triceps Extension** | Elbow position • ROM • Wrist position • Core tightness • Tempo |
| **9** | **Shrugs** | Shoulder path • ROM • Neck stability • Weight control • Pause at top |
| **10** | **Weighted Squats** | Stance • Depth • Knee tracking • Back posture • Weight distribution |
| **11** | **Seated Biceps Curls** | Elbow position • Shoulder stability • ROM • Wrist position • Back support |
| **12** | **Triceps Kickbacks** | Upper arm position • Elbow movement • Full extension • No torso swing • Neutral spine |
| **13** | **Rows** | Back angle • Elbow path • Shoulder retraction • Pulling motion • Stable base |
| **14** | **Deadlift** | Spine • Hips • Bar path • Lockout • Tempo |
| **15** | **Calf Raises** | ROM • Knee status • Balance • Tempo • Peak hold |

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.13.0 | Exercise recognition (MLP) |
| `torch` | ≥ 2.0.0 | Assessment models (CNN) |
| `mediapipe` | ≥ 0.10.0 | Pose extraction (33 landmarks) |
| `langgraph` | ≥ 0.0.30 | Coaching agent state machine |
| `langchain` | ≥ 0.1.0 | LLM orchestration |
| `google-generativeai` | ≥ 0.3.0 | Gemini API integration |
| `pydantic` | ≥ 2.0.0 | Data validation |
| `scikit-learn` | ≥ 1.3.0 | Model utilities, metrics |
| `joblib` | ≥ 1.3.0 | Assessment model serialization |
| `numpy` | ≥ 1.24.0 | Numerical computing |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `opencv-python` | ≥ 4.8.0 | Video processing |
| `matplotlib` | ≥ 3.7.0 | Visualization |

Full `requirements.txt` available at repository root.

---

## Testing

### Run Test Suite

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_experiment_1.py -v

# With output
pytest tests/test_experiment_1.py -v -s
```

### Test Files

| File | Purpose |
|------|---------|
| `test_experiment_1.py` | MLP training, single-run and multi-run validation |
| `test_05_small_cnn.py` | Legacy GEI model test (deprecated) |

### Example Test

```python
def test_experiment_1_single_run():
    """Verify single training run completes successfully."""
    from src.scripts.experiment_1 import train_experiment_1
    
    results = train_experiment_1(
        npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
        config_path='config/experiment_1_baseline_front.yaml',
        seed=99,
        max_epochs=5  # Quick test
    )
    
    assert results['val_accuracy'] > 0.0
    assert 'test_f1' in results
```

---

## Limitations & Future Work

### Current Limitations

1. **Assessment MAE ~3.5/10** — Relative high error on 0–10 scale; separating expert-level nuances remains challenging
2. **Dataset size** — 51 subjects is modest; performance varies across random splits (hence 30-run evaluation)
3. **Gender imbalance** — 80.4% male; limited female participant diversity
4. **Limited lower-body samples** — Dataset emphasizes upper-body exercises
5. **Occlusion sensitivity** — Hand landmarks (for supination features) degrade under arm occlusion or fast motion
6. **Single-camera deployment** — Current system processes front or side view; no view fusion

### Future Directions

1. **Improve assessment accuracy** — Explore larger models, self-supervised pretraining, or multimodal inputs (RGB + pose + IMU)
2. **Expand dataset** — Recruit more participants (especially female), add more diverse exercises and body types
3. **View fusion** — Combine complementary front + side kinematics for robust 3D form assessment
4. **On-device optimization** — TensorFlow Lite quantization for real-time mobile deployment
5. **Self-supervised learning** — Pre-train on unlabeled pose sequences to improve generalization with limited labeled data
6. **Personalization** — Adapt coaching feedback based on user's fitness level, goals, and preferences

---

## Citation

If you use this code or dataset, please cite:

**BibTeX** (paper):
```bibtex
@article{ahmed2026virtual,
  title={A Pose-Based AI Coach for Exercise Recognition and Quality Assessment},
  author={Ahmed, Aly and Eldomiatty, Al Amir and Elsayed, Zeyad and Gomaa, Walid},
  journal={[Publication venue TBD]},
  year={2026},
  note={Available at https://github.com/AhmedAly0/AI-Virtual-Coach}
}
```

**Plain text**:
```
Ahmed Aly, Al Amir Eldomiatty, Zeyad Elsayed, and Walid Gomaa. "A Pose-Based 
AI Coach for Exercise Recognition and Quality Assessment." 2026. 
https://github.com/AhmedAly0/AI-Virtual-Coach
```

**Dataset** (upon request):
```
The AI Virtual Coach Dataset (51 volunteers, 15 exercises, 308 videos) is 
available upon reasonable request. Please contact the authors.
```

---

## Acknowledgments

- **Funding**: STDF (Science and Technology Development Fund, Egypt), Project ID: 51399 — "VERAS: Virtual Exercise Recognition and Assessment System"
- **Institution**: Egypt-Japan University of Science and Technology (E-JUST), Department of Computer Science and Engineering
- **Coaches**: Annotation and validation provided by 2 certified fitness coaches
- **AI tools**: Portions of this project's documentation and code were drafted with assistance from Claude Opus 4.5, Claude Sonnet 4.5, Google Gemini 3 Pro, and ChatGPT 5.2

---

## License

**Status**: License file pending (TBD). The dataset is available upon reasonable request. Code will be released under an open-source license (MIT or Apache 2.0 recommended). Please contact the authors for permission to use the dataset in publications or commercial products.

---

## Authors / Contact

| Name | Institution | Email |
|------|------------|-------|
| Ahmed Mohamed Ahmed Ali | E-JUST | ahmed.mohammad@ejust.edu.eg |
| Zeyad Mohamed Mahmoud | E-JUST | zeyad.elsayed@ejust.edu.eg |
| Al Amir Hossam | E-JUST | alamir.eldomiatty@ejust.edu.eg |
| **Supervisor:** Prof. Walid Gomaa | E-JUST, Alexandria University | walid.gomaa@ejust.edu.eg |

---

## Quick Links

- **GitHub**: https://github.com/AhmedAly0/AI-Virtual-Coach
- **Research Paper**: [docs/Research_Paper_Draft/main.tex](docs/Research_Paper_Draft/main.tex)
- **Thesis**: [docs/Final_Thesis/main.tex](docs/Final_Thesis/main.tex)
- **API Contract**: [src/pipelines/backend_api_contract.md](src/pipelines/backend_api_contract.md)

---

**Last Updated**: February 8, 2026  
**Version**: 1.0 (Unified README)
