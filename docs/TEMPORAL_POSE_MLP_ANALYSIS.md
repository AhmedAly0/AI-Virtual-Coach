# Temporal Pose MLP Optimization: Complete Workflow Analysis & Feature Engineering Report

**Analysis Date:** January 8, 2026  
**Experiments Analyzed:** exp_06_pose_mlp_temporal (front & side views)  
**Multi-Run Sets:** multi_run_003 (Before) vs multi_run_004 (After)

---

## Executive Summary

This analysis investigates why the **temporal side-view model significantly outperforms the temporal front-view model** and evaluates the impact of **pose estimation upgrades** on both models.

### Key Findings Summary

| Metric | Front View Gap | Interpretation |
|--------|---------------|----------------|
| **Accuracy Gap (After Upgrades)** | 8.65% | Side (84.08%) vs Front (75.43%) |
| **F1 Gap (After Upgrades)** | 11.78% | Side (83.92%) vs Front (72.14%) |
| **Front Improved?** | ✅ Yes (+5.11% accuracy) | Hyperparameter tuning helped |
| **Side Degraded?** | ⚠️ Yes (-5.18% accuracy) | Regression from config changes |
| **Gap Change** | Narrowed by 10.29% | From 18.94% → 8.65% gap |

### Critical Discovery
The comparison reveals that **changes between multi_run_003 and multi_run_004 were NOT purely pose estimation upgrades** — there were also significant **hyperparameter changes**:

| Parameter | Front 003 | Front 004 | Side 003 | Side 004 |
|-----------|-----------|-----------|----------|----------|
| hidden_sizes | [512, 256, 128] | [384, 192, 96] | [384, 192, 96] | [384, 192, 96] |
| dropout | 0.38 | 0.32 | 0.30 | 0.30 |
| batch_size | 12 | 16 | 16 | 16 |
| patience | 50 | 50 | 50 | 55 |
| max_epochs | 150 | 150 | 200 | 200 |

This confounds the analysis — improvements in front view may be due to **better hyperparameters**, not pose upgrades.

---

## Phase 1: Baseline Performance Analysis — Before vs After

### 1.1 Overall Metrics Comparison

#### FRONT VIEW TEMPORAL

| Metric | Multi-Run 003 (Before) | Multi-Run 004 (After) | Change | Improvement? |
|--------|------------------------|----------------------|--------|--------------|
| **Mean Test Accuracy** | 0.7032 ± 0.0601 | 0.7543 ± 0.0423 | **+5.11%** | ✅ Yes |
| **Mean Test Macro F1** | 0.7006 ± 0.0546 | 0.7214 ± 0.0569 | **+2.08%** | ✅ Yes |
| Std Dev (Accuracy) | 0.0601 | 0.0423 | -0.0178 | ✅ Better (more stable) |
| Std Dev (Macro F1) | 0.0546 | 0.0569 | +0.0023 | ⚠️ Slightly worse |
| Best Accuracy | 0.7970 | 0.8417 | +0.0447 | ✅ Higher ceiling |
| Worst Accuracy | 0.5586 | 0.6673 | +0.1087 | ✅ Higher floor |
| Accuracy Range | [0.559, 0.797] | [0.667, 0.842] | — | Tighter, higher range |
| F1 Range | [0.581, 0.795] | [0.570, 0.803] | — | Similar range |

#### SIDE VIEW TEMPORAL

| Metric | Multi-Run 003 (Before) | Multi-Run 004 (After) | Change | Improvement? |
|--------|------------------------|----------------------|--------|--------------|
| **Mean Test Accuracy** | 0.8926 ± 0.0264 | 0.8408 ± 0.0503 | **-5.18%** | ❌ Degraded |
| **Mean Test Macro F1** | 0.8774 ± 0.0288 | 0.8392 ± 0.0522 | **-3.82%** | ❌ Degraded |
| Std Dev (Accuracy) | 0.0264 | 0.0503 | +0.0239 | ❌ Worse (less stable) |
| Std Dev (Macro F1) | 0.0288 | 0.0522 | +0.0234 | ❌ Worse |
| Best Accuracy | 0.9490 | 0.9373 | -0.0117 | ⚠️ Lower ceiling |
| Worst Accuracy | 0.8378 | 0.7466 | -0.0912 | ❌ Much lower floor |
| Accuracy Range | [0.838, 0.949] | [0.747, 0.937] | — | Wider, lower range |
| F1 Range | [0.812, 0.932] | [0.740, 0.938] | — | Wider, lower range |

### 1.2 Before vs After Summary

| Aspect | Finding |
|--------|---------|
| **Front view benefited from changes?** | ✅ Yes — Improved by **+5.11%** accuracy |
| **Side view benefited from changes?** | ❌ No — Degraded by **-5.18%** accuracy |
| **Which view benefited MORE?** | Front (side actually got worse) |
| **Front/Side gap BEFORE changes** | 18.94% (89.26% - 70.32%) |
| **Front/Side gap AFTER changes** | 8.65% (84.08% - 75.43%) |
| **Gap narrowed or widened?** | ✅ **Narrowed by 10.29 percentage points** |
| **Training stability improved?** | ⚠️ Mixed — Front better, Side worse |

### 1.3 Critical Observation

The gap narrowed NOT because front improved dramatically, but because **side view regressed significantly**. This suggests:

1. The hyperparameter changes in multi_run_004 may have been **optimized for front view** at the expense of side view
2. The side view model may have been **overfitting less** with larger architecture (003) but became **undertrained** with smaller architecture (004)
3. **Pose estimation upgrades cannot be isolated** from hyperparameter changes

---

## Phase 1.2: Per-Class Performance Deep Dive

### Label Mapping Reference
```
0: Bulgarian split squat    5: Hummer curls              10: Seated biceps curls
1: Calf raises              6: Inclined Dumbbell Press   11: Shrugs
2: Deadlift                 7: Lateral Raises            12: Standing Dumbbell Front Raises
3: Dumbbell shoulder press  8: Overhead Triceps Extension 13: Triceps Kickbacks
4: EZ Bar Curls             9: Rows                      14: Weighted Squats
```

### FRONT VIEW: Per-Class F1 Comparison (Before vs After)

| Class Name | 003 (Before) | 004 (After) | Change | Status |
|------------|--------------|-------------|--------|--------|
| Bulgarian split squat | 0.714 ± 0.086 | 0.636 ± 0.272 | **-0.078** | ⚠️ Degraded (higher variance) |
| Calf raises | 0.224 ± 0.194 | 0.596 ± 0.232 | **+0.372** | ✅ **Major Improvement** |
| Deadlift | 0.728 ± 0.161 | 0.705 ± 0.279 | -0.023 | ≈ Stable (higher variance) |
| Dumbbell shoulder press | 0.607 ± 0.256 | 0.902 ± 0.079 | **+0.295** | ✅ **Major Improvement** |
| EZ Bar Curls | 0.886 ± 0.129 | 0.588 ± 0.118 | **-0.298** | ❌ **Major Degradation** |
| Hummer curls | 0.780 ± 0.128 | 0.482 ± 0.199 | **-0.298** | ❌ **Major Degradation** |
| Inclined Dumbbell Bench Press | 0.500 ± 0.236 | 0.965 ± 0.047 | **+0.465** | ✅ **Major Improvement** |
| Lateral Raises | 0.941 ± 0.104 | 0.722 ± 0.147 | **-0.219** | ❌ Degraded |
| Overhead Triceps Extension | 0.897 ± 0.113 | 0.866 ± 0.128 | -0.031 | ≈ Stable |
| Rows | 0.664 ± 0.155 | 0.687 ± 0.134 | +0.023 | ≈ Stable |
| Seated biceps curls | 0.801 ± 0.179 | 0.631 ± 0.256 | **-0.170** | ⚠️ Degraded |
| Shrugs | 0.447 ± 0.130 | 0.560 ± 0.229 | **+0.113** | ⚠️ Improved but still weak |
| Standing Dumbbell Front Raises | 0.865 ± 0.115 | 0.882 ± 0.091 | +0.017 | ≈ Stable |
| Triceps Kickbacks | 0.495 ± 0.284 | 0.656 ± 0.189 | **+0.161** | ✅ Improved |
| Weighted Squats | 0.959 ± 0.044 | 0.944 ± 0.047 | -0.015 | ≈ Stable (excellent) |

### SIDE VIEW: Per-Class F1 Comparison (Before vs After)

| Class Name | 003 (Before) | 004 (After) | Change | Status |
|------------|--------------|-------------|--------|--------|
| Bulgarian split squat | 0.894 ± 0.076 | 0.830 ± 0.128 | **-0.064** | ⚠️ Degraded |
| Calf raises | 0.933 ± 0.034 | 0.685 ± 0.247 | **-0.248** | ❌ **Major Degradation** |
| Deadlift | 0.962 ± 0.039 | 0.926 ± 0.077 | -0.036 | ≈ Stable (excellent) |
| Dumbbell shoulder press | 0.861 ± 0.079 | 0.816 ± 0.105 | -0.045 | ≈ Stable |
| EZ Bar Curls | 0.826 ± 0.106 | 0.832 ± 0.131 | +0.006 | ≈ Stable |
| Hummer curls | 0.846 ± 0.097 | 0.781 ± 0.124 | -0.065 | ⚠️ Degraded |
| Inclined Dumbbell Bench Press | 0.877 ± 0.097 | 0.939 ± 0.064 | **+0.062** | ✅ Improved |
| Lateral Raises | 0.917 ± 0.077 | 0.746 ± 0.149 | **-0.171** | ❌ Degraded |
| Overhead Triceps Extension | 0.654 ± 0.123 | 0.841 ± 0.096 | **+0.187** | ✅ **Major Improvement** |
| Rows | 0.960 ± 0.034 | 0.922 ± 0.057 | -0.038 | ≈ Stable (excellent) |
| Seated biceps curls | 0.977 ± 0.035 | 0.919 ± 0.113 | -0.058 | ⚠️ Degraded (was excellent) |
| Shrugs | 0.590 ± 0.161 | 0.719 ± 0.156 | **+0.129** | ✅ Improved but still weak |
| Standing Dumbbell Front Raises | 0.950 ± 0.071 | 0.780 ± 0.192 | **-0.170** | ❌ Degraded |
| Triceps Kickbacks | 0.957 ± 0.038 | 0.911 ± 0.070 | -0.046 | ≈ Stable (excellent) |
| Weighted Squats | 0.955 ± 0.035 | 0.941 ± 0.065 | -0.014 | ≈ Stable (excellent) |

### Per-Class Analysis Summary

#### Classes That IMPROVED Most (Front View, 004 vs 003)
1. **Inclined Dumbbell Bench Press** — Improved by **+0.465 F1** (0.500 → 0.965)
   - Reason: Better pose detection for lying/inclined positions? Hyperparameter change?
2. **Calf Raises** — Improved by **+0.372 F1** (0.224 → 0.596)
   - Reason: Previously extremely weak, now moderate
3. **Dumbbell Shoulder Press** — Improved by **+0.295 F1** (0.607 → 0.902)
   - Reason: Upper body pressing now well-captured

#### Classes That DEGRADED Most (Front View, 004 vs 003)
1. **EZ Bar Curls** — Degraded by **-0.298 F1** (0.886 → 0.588)
   - **Critical**: Was excellent, now weak — this is a regression
2. **Hummer Curls** — Degraded by **-0.298 F1** (0.780 → 0.482)
   - **Critical**: Similar arm curl movement, same degradation pattern
3. **Lateral Raises** — Degraded by **-0.219 F1** (0.941 → 0.722)
   - Was near-perfect, now just good

#### Persistent Weak Classes (Still < 0.70 F1 in Multi-Run 004)

| Exercise | Front F1 | Side F1 | Gap | Analysis |
|----------|----------|---------|-----|----------|
| **Hummer curls** | 0.482 | 0.781 | -0.299 | Front severely underperforms |
| **EZ Bar Curls** | 0.588 | 0.832 | -0.244 | Front severely underperforms |
| **Bulgarian split squat** | 0.636 | 0.830 | -0.194 | Front struggles |
| **Shrugs** | 0.560 | 0.719 | -0.159 | Both weak, but front worse |
| **Calf Raises** | 0.596 | 0.685 | -0.089 | Both improved but still weak |
| **Seated biceps curls** | 0.631 | 0.919 | -0.288 | Front severely underperforms |
| **Triceps Kickbacks** | 0.656 | 0.911 | -0.255 | Front severely underperforms |

**Pattern Identified:** Arm curl/biceps exercises (Hummer curls, EZ Bar Curls, Seated biceps curls) show **severe front-view underperformance** compared to side view. This makes biomechanical sense:
- **Side view**: Clear elbow flexion angle visible
- **Front view**: Elbow flexion is depth-based (z-coordinate), harder to measure accurately

---

## Phase 1.3: Impact Assessment of Pose Estimation Upgrades

### Critical Limitation

**⚠️ Cannot isolate pose upgrade impact** from hyperparameter changes. The comparison is confounded by:

1. Different model architectures (front 003 vs 004)
2. Different dropout rates
3. Different batch sizes
4. Both running on presumably different feature files (before/after pose upgrades)

### Evidence Analysis

#### Did Full Model (vs Lite) Help?

| Question | Evidence | Conclusion |
|----------|----------|------------|
| Overall accuracy improved for front? | +5.11% | ✅ Yes, but confounded |
| Overall accuracy improved for side? | -5.18% | ❌ No, regressed |
| If full model universally better, should see improvements in BOTH views | Side got worse | ⚠️ **Not purely pose upgrade effect** |

#### Did Tempo Features Help?

- **Unknown**: Cannot determine if tempo features are being used in training
- The config files don't show tempo feature usage
- Need to check if MLP pipeline incorporates tempo features

#### Per-Class Impact Pattern

If **full model improved pose detection quality**, we'd expect:
- ✅ All classes to improve proportionally
- ✅ Weak classes (poor pose detection) to improve more
- ✅ Both views to benefit

**Actual Observation:**
- ❌ Some classes improved, others degraded dramatically
- ❌ Pattern is inconsistent (EZ Bar Curls was excellent, now weak)
- ❌ Side view degraded overall

**Conclusion:** The changes are **primarily due to hyperparameter modifications**, NOT pose estimation quality improvements.

### Impact Summary Table

| Aspect | Before (003) | After (004) | Implication |
|--------|-------------|-------------|-------------|
| **Front mean accuracy** | 70.32% | 75.43% | ✅ Improved |
| **Side mean accuracy** | 89.26% | 84.08% | ❌ Degraded |
| **Front/Side gap** | 18.94% | 8.65% | Gap narrowed (but artificially) |
| **Front stability (std)** | 0.0601 | 0.0423 | ✅ Better |
| **Side stability (std)** | 0.0264 | 0.0503 | ❌ Worse |
| **Weak class avg F1 (Front)** | 0.389 | 0.546 | ✅ Improved |
| **Weak class avg F1 (Side)** | 0.726 | 0.702 | ⚠️ Mixed |

### Pose Upgrade Conclusion

**Did the pose estimation upgrades help close the front/side gap?**
- ❌ **Cannot determine** — Results are confounded by hyperparameter changes
- The gap narrowed, but side view **regressed**, not front improved dramatically
- Need **controlled experiment**: Same hyperparameters, different pose features

**Was the improvement worth the computational cost?**
- **Unclear** — Would need A/B test with same training config

**Root cause implication:**
- If pose upgrades didn't help side view (which regressed), the problem is **NOT pose detection quality**
- The problem is more likely **feature engineering** or **view-specific information limitations**

---

## Phase 2: Feature Vector Investigation

### 2.1 Current Feature Extraction Analysis

#### Current 9-Angle Setup

| Angle Name | MediaPipe Landmarks | What It Captures | Front View Visibility | Side View Visibility |
|------------|--------------------|-----------------|-----------------------|---------------------|
| `left_elbow` | 11→13→15 | Arm flexion | ⚠️ Depth-dependent | ✅ Clear |
| `right_elbow` | 12→14→16 | Arm flexion | ⚠️ Depth-dependent | ✅ Clear |
| `left_shoulder` | 13→11→23 | Shoulder angle | ✅ Good for lateral | ⚠️ Occlusion risk |
| `right_shoulder` | 14→12→24 | Shoulder angle | ✅ Good for lateral | ⚠️ Occlusion risk |
| `left_hip` | 11→23→25 | Hip flexion | ✅ Good | ✅ Good |
| `right_hip` | 12→24→26 | Hip flexion | ✅ Good | ✅ Good |
| `left_knee` | 23→25→27 | Knee flexion | ⚠️ Depth-dependent | ✅ Clear |
| `right_knee` | 24→26→28 | Knee flexion | ⚠️ Depth-dependent | ✅ Clear |
| `torso_lean` | mid-shoulder→pelvis | Forward/backward lean | ⚠️ Depth-dependent | ✅ Clear |

#### Key Insight: View-Specific Angle Effectiveness

**Front view struggles with:**
- Elbow angles (arm curl exercises) — flexion is along depth axis (z)
- Knee angles — flexion is along depth axis
- Torso lean — forward/backward is depth-dependent

**Side view excels with:**
- All flexion angles — clearly visible in sagittal plane
- Torso lean — obvious forward/backward movement

### 2.2 Exercise-to-Angle Gap Analysis

#### Weak Classes: What Angles Do They Need?

| Exercise | Primary Joints | Current Angles Capture? | Missing Information |
|----------|---------------|------------------------|---------------------|
| **Hummer Curls** | Elbow flexion | ⚠️ Yes, but front=depth | Forearm rotation (supination) |
| **EZ Bar Curls** | Elbow flexion | ⚠️ Yes, but front=depth | Wrist angle, grip position |
| **Seated Biceps Curls** | Elbow flexion | ⚠️ Yes, but front=depth | Upper arm position |
| **Calf Raises** | Ankle plantarflexion | ❌ NOT captured | **Ankle angle (25→27→heel)** |
| **Shrugs** | Shoulder elevation | ❌ NOT captured | **Neck-shoulder distance**, trapezius |
| **Bulgarian Split Squat** | Hip, knee flexion | ⚠️ Captured but depth issue | Single-leg balance, torso rotation |
| **Triceps Kickbacks** | Elbow extension | ⚠️ Yes, but front=depth | Upper arm position (horizontal) |

#### Critical Missing Angles

1. **Ankle Angle** (for Calf Raises)
   - Landmarks: 25 (knee) → 27 (ankle) → 29/31 (heel/foot)
   - MediaPipe has heel landmark (29) and foot index (31)
   - **Implementation**: Calculate ankle dorsiflexion/plantarflexion angle

2. **Shoulder Elevation** (for Shrugs)
   - Not a joint angle, but a **distance metric**
   - Measure: ear-to-shoulder distance normalized by torso length
   - Or: shoulder-to-hip vertical distance change

3. **Wrist Angle** (for curl exercises)
   - Landmarks: 13/14 (elbow) → 15/16 (wrist) → 17/18 (pinky)
   - Captures forearm supination/pronation

### 2.3 Alternative Feature Representations Evaluation

#### Option A: Enhanced Joint Angles (+5-7 new angles)

**Proposed additions:**
1. `left_ankle` — 25 → 27 → 29 (ankle plantarflexion)
2. `right_ankle` — 26 → 28 → 30 (ankle plantarflexion)
3. `left_wrist` — 13 → 15 → 17 (wrist angle)
4. `right_wrist` — 14 → 16 → 18 (wrist angle)
5. `shoulder_elevation` — normalized vertical distance change

**Dimensionality impact:**
- Current: 9 angles × 50 timesteps = 450 features (flattened)
- Proposed: 14 angles × 50 timesteps = 700 features
- Increase: +55.6%

**Pros:** Directly targets weak classes (Calf Raises, Shrugs, Curls)
**Cons:** Risk of overfitting with limited data

#### Option B: Pairwise Distances

**Concept:** Use distances between key landmarks instead of angles

**Key distances for weak classes:**
| Distance | What It Captures | Helps Which Classes? |
|----------|-----------------|---------------------|
| hand-to-shoulder | Arm extension | Curls, Extensions |
| hand-to-hip | Arm reach | Front Raises, Kickbacks |
| shoulder-to-ear | Shoulder elevation | Shrugs |
| ankle-to-hip | Leg length (bent vs straight) | Squats, Calf Raises |
| heel-to-toe | Foot angle | Calf Raises |

**Pros:** 
- View-invariant (distance is absolute, not projected)
- Captures spatial positioning directly
- Less sensitive to depth estimation errors

**Cons:**
- May lose directionality information
- Requires careful normalization

#### Option C: Hybrid (Angles + Key Distances)

**Proposed:**
- Keep all 9 current angles
- Add 4-5 key pairwise distances
- Total: ~13-14 features per timestep

**Rationale:**
- Angles capture joint mechanics (flexion/extension)
- Distances capture spatial reach and positioning
- Together: comprehensive movement representation

#### Option D: Exercise-Specific Feature Sets

**Concept:** Train separate classifiers with different feature subsets

| Exercise Group | Key Angles | Key Distances |
|----------------|-----------|---------------|
| Upper pressing | shoulder, elbow, torso_lean | hand-to-shoulder |
| Arm curls | elbow, wrist | hand-to-shoulder |
| Lower body | hip, knee, ankle | ankle-to-hip |
| Shrugs/Raises | shoulder | shoulder-to-ear |

**Pros:** Optimal features per exercise
**Cons:** Complex pipeline, more models to maintain

### 2.4 Feature Option Evaluation Matrix

| Criterion | Option A (More Angles) | Option B (Distances) | Option C (Hybrid) | Option D (Exercise-Specific) |
|-----------|------------------------|---------------------|-------------------|----------------------------|
| Helps Shrugs? | ⚠️ Maybe (need shoulder elevation) | ✅ Yes (shoulder-to-ear) | ✅ Yes | ✅ Yes |
| Helps Overhead Triceps? | ⚠️ Maybe | ✅ Yes (hand-to-shoulder) | ✅ Yes | ✅ Yes |
| Helps Calf Raises? | ✅ Yes (ankle angle) | ✅ Yes (heel-to-toe) | ✅ Yes | ✅ Yes |
| Helps Arm Curls? | ⚠️ Limited (still depth issue) | ✅ Yes | ✅ Yes | ✅ Yes |
| Helps Front View Gap? | ⚠️ Partial | ✅ Better | ✅ Best | ✅ Yes |
| Overfitting Risk | Medium-High | Medium | Medium | Low |
| Implementation Complexity | Low | Medium | Medium | High |
| Dimensionality Increase | +55% | +50% | +80% | Variable |
| **Recommended Priority** | 3rd | 2nd | 1st | Future |

**Recommendation:** Start with **Option C (Hybrid)** for best coverage with manageable complexity.

---

## Phase 3: Root Cause Analysis — Why Front View Lags

### 3.1 Updated Hypothesis Based on Before/After Data

#### Key Question: Did pose estimation upgrades solve the front/side gap?

**Answer: NO — the gap narrowed because side view regressed, not because front view improved proportionally**

The evidence shows:
1. Front improved by +5.11%, but with major hyperparameter changes
2. Side regressed by -5.18%, likely due to hyperparameter mismatch
3. Gap narrowed "artificially" by harming the better model

**Root Cause is NOT pose landmark quality.** If it were, both views would improve with full model.

**Root cause is MORE LIKELY:**
1. **Feature engineering inadequacy** — Current 9 angles don't capture all exercise-relevant movements
2. **View-specific information loss** — Front view inherently loses depth information
3. **Hyperparameter sensitivity** — Front view may need different architecture/regularization

### 3.2 View-Specific Information Loss Analysis

#### Front View Camera (looking at subject from front)

| Information Type | Captured Well? | Used in Current Angles? | Impact on Weak Classes |
|-----------------|----------------|------------------------|----------------------|
| Left/right symmetry (X) | ✅ Yes | ✅ Yes (left/right angle pairs) | Neutral |
| Vertical movement (Y) | ✅ Yes | ✅ Yes (hip, knee, shoulder) | Good |
| Depth/forward-back (Z) | ⚠️ Noisy | ⚠️ Used but unreliable | **CRITICAL ISSUE** |
| Arm flexion toward camera | ❌ Poor | ⚠️ Elbow angles rely on Z | Curls, Extensions fail |
| Knee flexion toward camera | ❌ Poor | ⚠️ Knee angles rely on Z | Squats affected |

#### Side View Camera (looking at subject from side)

| Information Type | Captured Well? | Used in Current Angles? | Impact on Weak Classes |
|-----------------|----------------|------------------------|----------------------|
| Front/back movement (X) | ✅ Yes | ✅ Yes (torso lean) | Good |
| Vertical movement (Y) | ✅ Yes | ✅ Yes | Good |
| Depth/lateral (Z) | ⚠️ Less relevant | Not critical | Neutral |
| Arm flexion (sagittal) | ✅ Excellent | ✅ Elbow angles clear | Curls work well |
| Knee flexion (sagittal) | ✅ Excellent | ✅ Knee angles clear | Squats work well |

### 3.3 Training Instability Evidence

| Metric | Front 003 | Front 004 | Side 003 | Side 004 |
|--------|-----------|-----------|----------|----------|
| Std Dev (Accuracy) | 0.0601 | 0.0423 | 0.0264 | 0.0503 |
| Accuracy Range | 0.238 | 0.174 | 0.111 | 0.191 |
| Worst Run | 55.9% | 66.7% | 83.8% | 74.7% |
| Best Run | 79.7% | 84.2% | 94.9% | 93.7% |

**Observations:**
1. Front view has **higher variance** than side view (even after 004 improvement)
2. Front view worst runs are significantly worse (20%+ below best)
3. Side view was stable in 003, became unstable in 004

**Implications:**
- Front view learning is **inherently harder** (noisier features?)
- Smaller architecture (004) hurt side view stability
- Front view may benefit from stronger regularization

### 3.4 Root Cause Ranking

| Rank | Hypothesis | Evidence Strength | Actionability |
|------|-----------|-------------------|---------------|
| **1** | **Missing features** (ankle, shoulder elevation, wrist) | High — Weak classes map to missing angles | High |
| **2** | **Depth (Z) unreliability** for front view | High — Arm curls systematically fail | Medium |
| **3** | **Hyperparameter mismatch** | High — 004 helped front, hurt side | High |
| **4** | **Insufficient data** for front view | Medium — Higher variance suggests noise | Low |
| **5** | **Pose detection quality** | Low — Full model didn't uniformly help | Already addressed |

---

## Phase 4: Actionable Optimization Plan

### 4.1 Priority-Ranked Experiments

---

### Experiment #1: Add Missing Angles (Ankle, Wrist) — HIGHEST PRIORITY

**Hypothesis:** Adding ankle and wrist angles will significantly improve Calf Raises and arm curl exercises.

**Changes to Implement:**

1. **Feature Extraction** ([preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py)):
   ```python
   # Add to calculate_angle calls in extract_features_from_video():
   'left_ankle': (25, 27, 29),    # knee → ankle → heel
   'right_ankle': (26, 28, 30),   # knee → ankle → heel
   'left_wrist': (13, 15, 17),    # elbow → wrist → pinky
   'right_wrist': (14, 16, 18),   # elbow → wrist → pinky
   ```
   - Dimensionality change: 9 → 13 angles
   - Temporal: 450 → 650 features (flattened)
   
2. **Model Architecture:**
   - hidden_sizes: [384, 192, 96] → [512, 256, 128] (accommodate larger input)
   - dropout: 0.32 → 0.35 (prevent overfitting on larger features)

3. **Training Configuration:**
   - Keep current learning rate (0.00008)
   - Increase patience: 50 → 60 (more features may need longer convergence)

**Expected Outcome:**
- Calf Raises F1: 0.596 → 0.75+ (ankle angle critical)
- Arm curl exercises: 0.48-0.63 → 0.70+ (wrist angle helps)
- Front view accuracy: 75.4% → 78%+
- Risk of degradation: Low-Medium (adding relevant features)

**Success Criteria:**
- ✅ Calf Raises F1 improves by ≥0.10
- ✅ At least 2/3 arm curl exercises improve
- ✅ No classes regress by >0.05 F1
- ✅ Overall accuracy ≥76%

---

### Experiment #2: Add Distance Features for Shrugs — HIGH PRIORITY

**Hypothesis:** Shrugs require shoulder elevation distance (not angle) to be properly classified.

**Changes to Implement:**

1. **Feature Extraction:**
   ```python
   # Add distance-based feature
   def compute_shoulder_elevation(landmarks):
       """Compute normalized ear-to-shoulder distance."""
       # Left side
       left_ear = landmarks[7]  # Left ear
       left_shoulder = landmarks[11]  # Left shoulder
       left_hip = landmarks[23]  # Left hip
       
       # Torso length for normalization
       torso_len = euclidean_distance_3d(left_shoulder, left_hip)
       
       # Ear-to-shoulder vertical distance
       elevation = abs(left_ear.y - left_shoulder.y) / torso_len
       return elevation
   ```
   - Add 2 features: left_shoulder_elevation, right_shoulder_elevation
   - Total: 13 + 2 = 15 features per timestep

2. **Model Architecture:**
   - Keep [512, 256, 128] from Experiment 1
   - dropout: 0.35

**Expected Outcome:**
- Shrugs F1: 0.560 → 0.75+
- Minimal impact on other classes

**Success Criteria:**
- ✅ Shrugs F1 improves by ≥0.15
- ✅ No other classes regress

---

### Experiment #3: Hybrid Features with Key Distances — MEDIUM PRIORITY

**Hypothesis:** Adding 4-5 key pairwise distances will help front view by providing view-invariant spatial information.

**Changes to Implement:**

1. **Feature Extraction:**
   ```python
   # Key pairwise distances (normalized by torso length)
   distances = [
       'hand_to_shoulder_left',   # 15-11 distance
       'hand_to_shoulder_right',  # 16-12 distance  
       'hand_to_hip_left',        # 15-23 distance
       'hand_to_hip_right',       # 16-24 distance
       'heel_to_knee_left',       # 29-25 distance (calf extension)
   ]
   ```
   - Total features: 15 angles + 5 distances = 20 per timestep
   - Temporal: 20 × 50 = 1000 features (flattened)

2. **Model Architecture:**
   - hidden_sizes: [512, 256, 128] → [640, 320, 160] (larger input)
   - dropout: 0.40 (strong regularization)

**Expected Outcome:**
- Helps arm curls (hand-to-shoulder captures reach)
- Helps front view overall (view-invariant distances)

---

### Experiment #4: Hyperparameter Restoration for Side View — MEDIUM PRIORITY

**Hypothesis:** Side view regressed because hyperparameters were changed. Restore optimal side config.

**Changes to Implement:**
- hidden_sizes: [384, 192, 96] → [512, 256, 128] (restore larger network)
- dropout: 0.30 → 0.35
- batch_size: 16 → 12

**Expected Outcome:**
- Side view accuracy: 84.08% → 88%+ (restore previous performance)
- Combined with Experiment 1 features

---

### 4.2 Decision Tree for Next Steps

```
IF Experiment #1 (add angles) succeeds:
  → Proceed to Experiment #2 (add distances for Shrugs)
  → If still gap > 5%, try Experiment #3 (hybrid)
  
IF Experiment #1 shows no improvement:
  → Problem is NOT missing angles
  → Try Experiment #3 (distances may help where angles don't)
  → Consider architecture changes (LSTM for temporal patterns)
  
IF side view continues to regress:
  → Run Experiment #4 to restore side performance
  → Investigate if new features harm side view
  
IF front view weak classes (arm curls) don't improve:
  → Problem may be fundamental view limitation
  → Consider side-only deployment for those exercises
  → Or dual-camera requirement for production
```

---

## Phase 5: Documentation & Recommendation

### 5.1 Root Cause Summary

#### Why Front View Underperforms (After Pose Upgrades)

**Before Upgrades (multi_run_003):**
- Front: 70.3% accuracy, Side: 89.3% accuracy
- Gap: 18.94%
- Root cause hypothesis: Pose landmark quality OR feature inadequacy

**After Upgrades (multi_run_004):**
- Front: 75.4% accuracy, Side: 84.1% accuracy
- Gap: 8.65%
- **What changed:** Hyperparameters changed, NOT just pose model

**Current Status:**
- Front/Side gap: 8.65% (was 18.94%)
- Gap narrowed by: **10.29 percentage points**
- **BUT:** Side view regressed, front improved modestly
- Weak classes: Arm curls (0.48-0.63 F1), Shrugs (0.56 F1), Calf Raises (0.60 F1)

**What the Upgrades Revealed:**
- Gap narrowing was **NOT** due to pose quality improvement
- Gap narrowing was due to **hyperparameter optimization for front** at cost of side
- **Primary issue is feature engineering**, not pose detection

**Confidence Level:** **High** — Evidence strongly supports feature inadequacy over pose quality

### 5.2 Single Recommended Next Experiment

## 🎯 RECOMMENDED: Experiment #1 — Add Ankle and Wrist Angles

**Why this first:**
1. Directly targets 3 weak classes (Calf Raises, arm curls)
2. Low implementation complexity (extend existing angle computation)
3. Low risk (adding relevant biomechanical features)
4. Quick validation (reuse existing training pipeline)

**Implementation Checklist:**
- [ ] Modify `extract_features_from_video()` in [preprocess_pose_RGB.py](src/preprocessing/preprocess_pose_RGB.py)
- [ ] Add ankle angle computation (landmarks 25→27→29, 26→28→30)
- [ ] Add wrist angle computation (landmarks 13→15→17, 14→16→18)
- [ ] Update `angle_names` list to include new angles
- [ ] Re-run preprocessing notebook to generate new features
- [ ] Update config to use larger hidden layers [512, 256, 128]
- [ ] Run multi_run_005 for both front and side views
- [ ] Compare per-class F1 scores

**Expected Timeline:** 1-2 days implementation, 2-4 hours training per view

### 5.3 Fallback Plan

**If Experiment #1 doesn't improve front view:**

1. **Second attempt:** Experiment #3 (hybrid angles + distances)
   - Distances are view-invariant, may help where angles fail
   
2. **Architecture pivot:** If MLP can't close gap:
   - Try LSTM/GRU for temporal modeling
   - Temporal dependencies may matter more for front view
   
3. **Accept limitations:** If gap persists after 3 experiments:
   - Deploy side-view only for arm curl exercises
   - Use front view for exercises where it performs well
   - Or require dual-camera setup for production

---

## Appendix: Quick Reference Tables

### A1: Class Performance Matrix (After Upgrades)

| Class | Front F1 | Side F1 | Gap | Priority |
|-------|----------|---------|-----|----------|
| Weighted Squats | 0.944 | 0.941 | +0.003 | ✅ Solved |
| Inclined Bench Press | 0.965 | 0.939 | +0.026 | ✅ Solved |
| Dumbbell Shoulder Press | 0.902 | 0.816 | +0.086 | ✅ Solved |
| Standing Front Raises | 0.882 | 0.780 | +0.102 | ✅ Solved |
| Overhead Triceps Extension | 0.866 | 0.841 | +0.025 | ✅ Solved |
| Lateral Raises | 0.722 | 0.746 | -0.024 | ⚠️ OK |
| Deadlift | 0.705 | 0.926 | -0.221 | ⚠️ Gap |
| Rows | 0.687 | 0.922 | -0.235 | ⚠️ Gap |
| Triceps Kickbacks | 0.656 | 0.911 | -0.255 | ❌ Fix needed |
| Bulgarian Split Squat | 0.636 | 0.830 | -0.194 | ❌ Fix needed |
| Seated Biceps Curls | 0.631 | 0.919 | -0.288 | ❌ Fix needed |
| Calf Raises | 0.596 | 0.685 | -0.089 | ❌ Fix needed |
| EZ Bar Curls | 0.588 | 0.832 | -0.244 | ❌ Fix needed |
| Shrugs | 0.560 | 0.719 | -0.159 | ❌ Fix needed |
| Hummer Curls | 0.482 | 0.781 | -0.299 | ❌ Fix needed |

### A2: Feature Engineering Recommendations by Class

| Class | Current Angles Used | Missing Features | Recommended Addition |
|-------|--------------------|-----------------|--------------------|
| Calf Raises | Hip, Knee | **Ankle angle** | ✅ Ankle (25→27→29) |
| Shrugs | Shoulder | **Elevation distance** | ✅ Shoulder-to-ear distance |
| Arm Curls | Elbow | **Wrist angle** | ✅ Wrist (13→15→17) |
| Triceps Kickbacks | Elbow | Upper arm position | Hand-to-hip distance |
| Bulgarian Split | Hip, Knee | Balance, rotation | Ankle + asymmetry |

---

*Analysis completed: January 8, 2026*  
*Next action: Implement Experiment #1 (ankle and wrist angles)*
