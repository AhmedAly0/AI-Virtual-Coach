# AI Virtual Coach — Project Structure

## 🎯 Project Overview

This project implements an **AI-powered virtual fitness coach** that combines:

1. **Exercise Recognition** — Classify which exercise a person is performing from MediaPipe pose landmarks using a temporal MLP.
2. **Form Assessment** — Score repetition quality per exercise using per-exercise ML models.
3. **Coaching Agent** — Generate personalized post-exercise feedback via a LangGraph agent backed by Gemini LLM.

The system is designed for a **mobile (Flutter) front-end** that streams pose data to a backend API.

---

## 📁 Project Structure

```
ai-virtual-coach/
├── config/                              # Experiment YAML configs
│   ├── experiment_1_baseline_front.yaml     # 19-feature MLP, front view
│   ├── experiment_1_baseline_side.yaml      # 19-feature MLP, side view
│   ├── experiment_1_specialized_front.yaml  # 37-feature MLP, front view
│   ├── experiment_1_specialized_side.yaml   # 37-feature MLP, side view
│   └── archive/                             # Legacy GEI configs (deprecated)
│
├── datasets/
│   ├── Clips/                           # Raw exercise video clips (15 exercises)
│   ├── GEIs_of_rgb_front/              # Front-view Gait Energy Images (legacy)
│   ├── GEIs_of_rgb_side/               # Side-view Gait Energy Images (legacy)
│   ├── Mediapipe pose estimates/        # ★ Active NPZ data
│   │   ├── pose_data_front_19_features.npz
│   │   └── pose_data_side_19_features.npz
│   ├── old pose estimates/              # Legacy NPZ files
│   ├── Annotation aspects.txt           # Per-exercise assessment criteria
│   ├── metadata.xlsx.csv                # Subject demographics
│   ├── pose_landmarker_full.task        # MediaPipe model (full)
│   └── pose_landmarker_lite.task        # MediaPipe model (lite)
│
├── src/                                 # ★ Source code
│   ├── agents/                          # Coaching feedback agent
│   │   ├── coaching_agent.py            # LangGraph workflow (Gemini LLM)
│   │   ├── config.py                    # Agent configuration & API keys
│   │   ├── exercise_criteria.py         # Per-exercise assessment criteria
│   │   ├── prompts.py                   # LLM prompt templates
│   │   └── state.py                     # Pydantic state models
│   │
│   ├── data/                            # Data loading & preprocessing
│   │   ├── data_loader.py              # load_pose_data, load_pose_enhanced_data, split_by_subject
│   │   ├── preprocessing.py            # prep_tensors, resize, normalize
│   │   ├── augmentation.py             # Data augmentation pipelines
│   │   └── dataset_builder.py          # tf.data.Dataset builders
│   │
│   ├── models/                          # Model architectures
│   │   ├── model_builder.py            # build_model_*, BACKBONE_REGISTRY
│   │   └── assessment_models/          # 30 serialized .joblib files (15 exercises × model + scaler)
│   │
│   ├── scripts/                         # Training scripts
│   │   ├── experiment_1.py             # ★ Pose MLP training (single + multi-run)
│   │   ├── vc_core.py                  # Video → assessment pipeline utilities
│   │   ├── video_to_assessment_cnn_all.py  # End-to-end video assessment (PyTorch)
│   │   ├── gei_embeddings.py           # GEI embedding utilities (ArcFace)
│   │   ├── gei_grid_search_utils.py    # GEI grid-search helpers
│   │   └── archive/                    # Archived GEI experiment scripts (deprecated)
│   │
│   ├── pipelines/                       # API contracts & orchestration
│   │   └── backend_api_contract.md     # Flutter ↔ Backend API specification
│   │
│   ├── preprocessing/                   # Raw data preprocessing
│   │   ├── preprocess_pose_RGB.py      # MediaPipe pose extraction from videos
│   │   └── analyze_frame_distribution.py  # Frame/FPS/duration analysis
│   │
│   └── utils/                           # Shared utilities
│       ├── io_utils.py                 # Folder management, seed setting
│       ├── metrics.py                  # Experiment tracking, parameter counting
│       └── visualization.py            # Training curves, confusion matrices, comparisons
│
├── notebooks/
│   ├── EDA/
│   │   └── Analysis.ipynb               # Exploratory data analysis
│   ├── exer_recog/
│   │   ├── 01_pose_mlp.ipynb            # ★ Main experiment notebook
│   │   ├── 99_comparison.ipynb          # Cross-experiment comparison
│   │   └── archive/                     # Archived GEI notebooks
│   └── pose_preprocessing/
│       ├── 00_pose_preprocessing.ipynb           # Pose feature extraction
│       └── 00b_frame_distribution_analysis.ipynb # Frame distribution analysis
│
├── output/
│   ├── exer_recog/
│   │   ├── exp_01_pose_mlp_baseline/    # ★ 19-feature results (front/ + side/)
│   │   ├── exp_01_pose_mlp_specialized/ # ★ 37-feature results (front/ + side/)
│   │   ├── exp_01_baseline/             # Legacy GEI baseline results
│   │   ├── exp_02_progressive/          # Legacy GEI progressive results
│   │   ├── exp_03_smart_heads/          # Legacy GEI smart-heads results
│   │   ├── exp_04_regularized/          # Legacy GEI regularized results
│   │   └── exp_05_small_cnn/            # Legacy GEI small-CNN results
│   └── assessment/
│       ├── outputs_front/               # Front-view assessment outputs
│       └── outputs_side/                # Side-view assessment outputs
│
├── tests/
│   ├── test_experiment_1.py             # Pose MLP validation tests
│   └── test_05_small_cnn.py             # Legacy GEI test
│
├── docs/
│   ├── EXPERIMENT_1_QUICK_REFERENCE.md  # Experiment 1 usage guide
│   ├── FEATURE_ENGINEERING.md           # Feature engineering documentation
│   ├── SUBJECT_WISE_SPLITTING_METHODOLOGY.md
│   ├── COACHING_AGENT_DOCUMENTATION.md  # Coaching agent architecture
│   ├── REFACTORING_SUMMARY.md
│   ├── PROJECT_STRUCTURE.md             # ← This file
│   ├── Research_Paper_Draft/            # Paper drafts
│   └── archive/                         # Archived docs
│
└── plots/                               # Static plots & figures
```

---

## 🔬 Experiment 1: Pose-Based MLP

**The sole active experiment** for exercise recognition.

| Item | Location |
|------|----------|
| Training script | `src/scripts/experiment_1.py` |
| Configs (baseline) | `config/experiment_1_baseline_{front,side}.yaml` |
| Configs (specialized) | `config/experiment_1_specialized_{front,side}.yaml` |
| Notebook | `notebooks/exer_recog/01_pose_mlp.ipynb` |
| Results (baseline) | `output/exer_recog/exp_01_pose_mlp_baseline/` |
| Results (specialized) | `output/exer_recog/exp_01_pose_mlp_specialized/` |

### Feature Sets

| Variant | Features | Description |
|---------|----------|-------------|
| **Baseline** | 19 per timestep | 13 joint angles + 6 pairwise distances |
| **Specialized** | 37 per timestep | 19 base + 18 exercise-specific discrimination features |

### Architecture

- **Input**: Flattened temporal features (T × F) → dense vector
- **MLP**: 3 hidden layers `[512, 256, 128]` with BatchNorm + Dropout
- **Output**: 15-class softmax
- **Multi-run**: 30 seeds for statistical robustness

### Training Functions

```python
from src.scripts import train_experiment_1, train_experiment_1_multi_run

# Single run
results = train_experiment_1(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml'
)

# Multi-run (30 seeds)
all_runs, stats = train_experiment_1_multi_run(
    npz_path='datasets/Mediapipe pose estimates/pose_data_front_19_features.npz',
    config_path='config/experiment_1_baseline_front.yaml'
)
```

---

## 🏋️ Exercise Assessment Pipeline

The assessment pipeline scores individual repetitions of a recognized exercise:

1. **Video → Pose**: MediaPipe extracts 33 3D landmarks per frame (`src/preprocessing/preprocess_pose_RGB.py`)
2. **Feature Engineering**: Joint angles, distances, and specialized features computed from landmarks
3. **Per-Exercise Models**: 15 scikit-learn models (`src/models/assessment_models/*.joblib`) score rep quality
4. **Coaching Agent**: LangGraph workflow generates natural-language feedback (`src/agents/coaching_agent.py`)

### API Contract

Defined in `src/pipelines/backend_api_contract.md`:
- `POST /api/session/analyze` — receives pose sequence, returns recognition + assessment + coaching feedback

---

## 🤖 Coaching Agent (LangGraph)

| Module | Purpose |
|--------|---------|
| `coaching_agent.py` | Stateful LangGraph workflow — loads criteria → analyzes scores → LLM feedback |
| `config.py` | API keys, model config, temperature settings |
| `exercise_criteria.py` | Maps 15 exercise IDs → assessment criteria strings |
| `prompts.py` | `ChatPromptTemplate` for Gemini |
| `state.py` | Pydantic state models (`AgentState`, `RepScore`, etc.) |

---

## 🔧 Module Reference

### src/data/

| Module | Key Exports |
|--------|-------------|
| `data_loader.py` | `load_data`, `load_pose_data`, `load_pose_enhanced_data`, `split_by_subject_two_way`, `split_by_subjects_three_way`, `build_subject_folds`, `verify_subject_split_integrity` |
| `preprocessing.py` | `prep_tensors`, `prep_tensors_with_preprocess`, `prep_tensors_grayscale` |
| `augmentation.py` | `data_augmentater`, `BASIC_AUGMENTATION`, `ENHANCED_AUGMENTATION` |
| `dataset_builder.py` | `build_datasets`, `build_datasets_three_way`, `build_pose_datasets_three_way`, `build_streaming_dataset` |

### src/models/

| Module | Key Exports |
|--------|-------------|
| `model_builder.py` | `build_model` (CNN), `build_model_for_backbone` / `_v2` / `_v3` / `_v4` (transfer learning), `get_callbacks`, `BACKBONE_REGISTRY` |
| `assessment_models/` | 30 `.joblib` files — per-exercise model + scaler pairs |

### src/scripts/

| Module | Purpose |
|--------|---------|
| `experiment_1.py` | `train_experiment_1()`, `train_experiment_1_multi_run()` — Pose MLP |
| `vc_core.py` | Video → assessment pipeline utilities |
| `video_to_assessment_cnn_all.py` | End-to-end video → per-rep assessment (PyTorch) |
| `gei_embeddings.py` | GEI embedding extraction (ArcFace) |
| `gei_grid_search_utils.py` | Grid-search helpers for GEI models |
| `archive/` | 5 deprecated GEI experiment scripts |

### src/utils/

| Module | Key Exports |
|--------|-------------|
| `io_utils.py` | `set_global_seed`, `create_results_folder`, `increment_run_folder` |
| `metrics.py` | `ExperimentTracker`, `get_all_model_parameters`, `log_fold_results` |
| `visualization.py` | `plot_training_curves`, `plot_confusion_matrix`, `create_comprehensive_comparison` |

### src/preprocessing/

| Module | Purpose |
|--------|---------|
| `preprocess_pose_RGB.py` | MediaPipe 3D pose extraction from video, joint angle computation |
| `analyze_frame_distribution.py` | Frame count / FPS / duration statistics for T_fixed selection |

---

## 📊 Supported Exercises (15 classes)

| ID | Exercise |
|----|----------|
| 1 | Dumbbell Shoulder Press |
| 2 | Hammer Curls |
| 3 | Standing Dumbbell Front Raises |
| 4 | Lateral Raises |
| 5 | Bulgarian Split Squat |
| 6 | EZ Bar Curls |
| 7 | Inclined Dumbbell Bench Press |
| 8 | Overhead Triceps Extension |
| 9 | Shrugs |
| 10 | Weighted Squats |
| 11 | Seated Biceps Curls |
| 12 | Triceps Kickbacks |
| 13 | Rows |
| 14 | Deadlift |
| 15 | Calf Raises |

---

## 📦 Key Dependencies

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
mediapipe>=0.10.0
langchain / langgraph      # Coaching agent
google-generativeai        # Gemini LLM
pydantic>=2.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0.0
joblib>=1.3.0
```

---

## 📄 Configuration

Each YAML config controls a single experiment variant:

```yaml
# config/experiment_1_baseline_front.yaml (example)
data:
  npz_path: datasets/Mediapipe pose estimates/pose_data_front_19_features.npz
  feature_type: all          # 19 features (13 angles + 6 distances)

model:
  hidden_layers: [512, 256, 128]
  dropout: 0.35
  num_classes: 15

training:
  batch_size: 16
  learning_rate: 0.00006
  max_epochs: 200

callbacks:
  early_stopping_patience: 60
  reduce_lr_patience: 15

multi_run:
  num_runs: 30
  base_seed: 42

results:
  base_dir: output/exer_recog/exp_01_pose_mlp_baseline/front
```

---

## 📝 Archive

Legacy GEI-based experiments (1–5) are archived but preserved for reference:

| Location | Contents |
|----------|----------|
| `src/scripts/archive/` | 5 GEI training scripts (with deprecation warnings) |
| `config/archive/` | 7 GEI YAML configs |
| `notebooks/exer_recog/archive/` | 5 GEI notebooks |
| `docs/archive/` | Historical refactoring docs |
| `output/exer_recog/exp_01_baseline/` … `exp_05_small_cnn/` | Legacy GEI results |

---

## 👤 Author
Ahmed Mohamed Ahmed

**Last Updated:** February 8, 2026
