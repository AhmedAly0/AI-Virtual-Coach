# Archived GEI-Based Experiments

> **⚠️ DEPRECATED** — These experiments are preserved for historical reference only.

## Overview

These experiments used **Gait Energy Images (GEIs)** as input for exercise recognition and have been fully superseded by **pose-based models** which achieve better performance with simpler input pipelines.

## Archived Experiments

| File | Original Name | Description |
|------|---------------|-------------|
| `experiment_1_gei_baseline.py` | exercise_recognition.py | 2-phase transfer learning on GEIs |
| `experiment_2_gei_progressive.py` | experiment_2.py | Progressive unfreezing on GEIs |
| `experiment_3_gei_smart_heads.py` | experiment_3.py | Architecture-specific classification heads on GEIs |
| `experiment_4_gei_regularized.py` | experiment_4.py | Regularized dual-pooling heads on GEIs |
| `experiment_5_gei_small_cnn.py` | experiment_5.py | Custom small CNN with subject-wise k-fold on GEIs |

## Replacement

Use the active pose-based experiments instead:

- **`src/scripts/exercise_recognition.py`** — Pose MLP (temporal features, 30-run evaluation)

## When Were These Archived?

Archived on **2026-02-08** as part of the experiment reorganization that promoted the pose-based MLP to Experiment 1.
