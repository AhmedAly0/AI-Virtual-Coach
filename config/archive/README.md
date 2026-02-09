# Archived Configuration Files

> **⚠️ DEPRECATED** — These configs are preserved for historical reference only.

## GEI-Based Experiment Configs

| File | Original Name | Description |
|------|---------------|-------------|
| `experiment_1_gei_baseline.yaml` | experiment_1.yaml | GEI transfer learning config |
| `experiment_2_gei_progressive.yaml` | experiment_2.yaml | GEI progressive unfreezing config |
| `experiment_3_gei_smart_heads.yaml` | experiment_3.yaml | GEI architecture-specific heads config |
| `experiment_4_gei_regularized.yaml` | experiment_4.yaml | GEI regularized heads config |
| `experiment_5_gei_small_cnn.yaml` | experiment_5.yaml | GEI small CNN config |
| `experiment_5_gei_multirun.yaml` | experiment_5_multirun.yaml | GEI small CNN multi-run config |

## Other Archived Configs

| File | Original Name | Description |
|------|---------------|-------------|
| `experiment_6_phase2_front.yaml` | experiment_6_phase2_front.yaml | Pose MLP phase 2 extended (deprecated variant) |

## Replacement

Use the active pose-based configs instead:

- `config/exer_recog_baseline_front.yaml` — Pose MLP baseline (front view)
- `config/exer_recog_baseline_side.yaml` — Pose MLP baseline (side view)
- `config/exer_recog_specialized_front.yaml` — Pose MLP specialized features (front view)
- `config/exer_recog_specialized_side.yaml` — Pose MLP specialized features (side view)
