"""Tests for the 37-feature quality assessment pipeline.

Covers:
  - Feature extraction shape and consistency
  - Landmark normalization
  - Rep segmentation stability (37 vs legacy 9)
  - CNN model forward pass
  - YAML config loading
  - Backward compatibility / deprecation warnings
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.quality_assessment import (
    compute_assessment_features,
    get_assessment_feature_names,
    normalize_landmarks_array,
    normalize_landmarks_sequence,
    extract_9_features,
    segment_reps_from_sequence,
    validate_feature_dimensions,
    _get_rep_seg_config,
    _load_rep_segmentation_config,
    build_cnn_subject_regressor,
    _attention_pool_reps,
    FEATURE_DIM,
    LEGACY_FEATURE_DIM,
)
from src.preprocessing.preprocess_pose_RGB import (
    compute_all_features_from_landmarks,
    compute_specialized_features,
    compute_side_specialized_features,
    ALL_FEATURE_NAMES,
    SPECIALIZED_FEATURE_NAMES,
    SIDE_SPECIALIZED_FEATURE_NAMES,
    ALL_EXTENDED_FEATURE_NAMES,
    SIDE_ALL_EXTENDED_FEATURE_NAMES,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_synthetic_landmarks(n_frames: int = 100, seed: int = 42) -> np.ndarray:
    """Create synthetic (N, 33, 3) landmark data that survives normalization.

    Landmarks are arranged so that the pelvis and shoulders are well-separated.
    """
    rng = np.random.RandomState(seed)
    lm = rng.randn(n_frames, 33, 3).astype(np.float32) * 0.1

    # Set shoulders and hips to known positions so normalization works
    for i in range(n_frames):
        lm[i, 11] = [0.4, 0.3, 0.0]   # left shoulder
        lm[i, 12] = [0.6, 0.3, 0.0]   # right shoulder
        lm[i, 23] = [0.4, 0.6, 0.0]   # left hip
        lm[i, 24] = [0.6, 0.6, 0.0]   # right hip
        # Add slight temporal variation for rep signal
        offset = 0.05 * np.sin(2 * np.pi * i / 20)
        lm[i, 15, 1] += offset  # left wrist y
        lm[i, 16, 1] += offset  # right wrist y
        lm[i, 13, 1] += offset * 0.5  # left elbow y
        lm[i, 14, 1] += offset * 0.5  # right elbow y

    return lm


# ============================================================================
# Test: Landmark Normalization
# ============================================================================

class TestLandmarkNormalization:

    def test_normalize_single_frame(self):
        lm = _make_synthetic_landmarks(1)[0]  # (33, 3)
        normed = normalize_landmarks_array(lm)
        assert normed is not None
        assert normed.shape == (33, 3)

        # Pelvis should be at origin
        pelvis = (normed[23] + normed[24]) / 2.0
        np.testing.assert_allclose(pelvis, 0.0, atol=1e-5)

        # Torso length should be 1.0
        mid_shoulder = (normed[11] + normed[12]) / 2.0
        pelvis_center = (normed[23] + normed[24]) / 2.0
        torso_len = np.linalg.norm(mid_shoulder - pelvis_center)
        np.testing.assert_allclose(torso_len, 1.0, atol=1e-5)

    def test_normalize_sequence(self):
        lm = _make_synthetic_landmarks(50)
        normed = normalize_landmarks_sequence(lm)
        assert normed.shape == (50, 33, 3)

    def test_normalize_degenerate(self):
        """When shoulders == hips, normalization should return None."""
        lm = np.zeros((33, 3), dtype=np.float32)
        result = normalize_landmarks_array(lm)
        assert result is None


# ============================================================================
# Test: Feature Extraction
# ============================================================================

class TestFeatureExtraction:

    def test_compute_assessment_features_front_37(self):
        lm = _make_synthetic_landmarks(30)
        normed = normalize_landmarks_sequence(lm)
        feats = compute_assessment_features(normed, view="front", feature_type="all_extended")
        assert feats.shape == (30, 37)
        assert feats.dtype == np.float32

    def test_compute_assessment_features_side_37(self):
        lm = _make_synthetic_landmarks(30)
        normed = normalize_landmarks_sequence(lm)
        feats = compute_assessment_features(normed, view="side", feature_type="all_extended")
        assert feats.shape == (30, 37)

    def test_compute_assessment_features_base_19(self):
        lm = _make_synthetic_landmarks(30)
        normed = normalize_landmarks_sequence(lm)
        feats = compute_assessment_features(normed, view="front", feature_type="base")
        assert feats.shape == (30, 19)

    def test_feature_names_match_dimensions(self):
        names_front = get_assessment_feature_names("front", "all_extended")
        assert len(names_front) == 37

        names_side = get_assessment_feature_names("side", "all_extended")
        assert len(names_side) == 37

        names_base = get_assessment_feature_names("front", "base")
        assert len(names_base) == 19

    def test_consistency_with_preprocess_module(self):
        """Verify that compute_assessment_features matches direct calls."""
        lm = _make_synthetic_landmarks(5)
        normed = normalize_landmarks_sequence(lm)

        feats = compute_assessment_features(normed, view="front", feature_type="all_extended")

        for i in range(5):
            base = compute_all_features_from_landmarks(normed[i])
            spec = compute_specialized_features(normed[i])
            expected = np.concatenate([base, spec])
            np.testing.assert_allclose(feats[i], expected, atol=1e-5)

    def test_invalid_view_raises(self):
        lm = _make_synthetic_landmarks(5)
        normed = normalize_landmarks_sequence(lm)
        with pytest.raises(ValueError, match="Invalid view"):
            compute_assessment_features(normed, view="top")

    def test_invalid_feature_type_raises(self):
        lm = _make_synthetic_landmarks(5)
        normed = normalize_landmarks_sequence(lm)
        with pytest.raises(ValueError, match="Invalid feature_type"):
            compute_assessment_features(normed, view="front", feature_type="bogus")


# ============================================================================
# Test: Legacy 9-Feature Deprecation
# ============================================================================

class TestLegacyDeprecation:

    def test_extract_9_features_deprecation_warning(self):
        lm = _make_synthetic_landmarks(1)[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            feats = extract_9_features(lm)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert feats.shape == (9,)

    def test_segment_reps_legacy_mode_warning(self):
        lm = _make_synthetic_landmarks(200)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reps, peaks, signal, used = segment_reps_from_sequence(
                exercise="Hummer curls",
                view="front",
                lm_seq_xyz=lm,
                feature_mode="9",
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

        assert used["feature_mode"] == "9"
        if reps.shape[0] > 0:
            assert reps.shape[2] == 9  # Legacy 9 features


# ============================================================================
# Test: Rep Segmentation (37-Feature Mode)
# ============================================================================

class TestRepSegmentation:

    def test_segment_produces_37_features(self):
        lm = _make_synthetic_landmarks(200)
        reps, peaks, signal, used = segment_reps_from_sequence(
            exercise="Hummer curls",
            view="front",
            lm_seq_xyz=lm,
            feature_mode="37",
        )
        assert used["feature_mode"] == "37"
        if reps.shape[0] > 0:
            assert reps.shape[2] == 37

    def test_segment_returns_valid_shapes(self):
        lm = _make_synthetic_landmarks(200)
        reps, peaks, signal, used = segment_reps_from_sequence(
            exercise="Lateral Raises",
            view="front",
            lm_seq_xyz=lm,
            feature_mode="37",
        )
        assert signal.shape == (200,)
        assert isinstance(peaks, list)
        assert "win" in used

    def test_segment_invalid_landmarks_raises(self):
        bad = np.zeros((10, 5, 3))  # Only 5 landmarks instead of 33
        with pytest.raises(ValueError, match="33 landmarks"):
            segment_reps_from_sequence("Rows", "front", bad, feature_mode="37")


# ============================================================================
# Test: CNN Model Architecture
# ============================================================================

class TestCNNModel:

    def test_forward_pass_37_features(self):
        model = build_cnn_subject_regressor(in_feats=37, n_aspects=5, T_fixed=50)
        x = np.random.randn(2, 50, 37).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (2, 5)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output

    def test_forward_pass_single_sample(self):
        model = build_cnn_subject_regressor(in_feats=37, n_aspects=3, T_fixed=50)
        x = np.random.randn(1, 50, 37).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (1, 3)

    def test_model_attributes(self):
        model = build_cnn_subject_regressor(in_feats=37, n_aspects=5, T_fixed=50)
        # Input shape: (None, 50, 37), output shape: (None, 5)
        assert model.input_shape == (None, 50, 37)
        assert model.output_shape == (None, 5)

    def test_attention_pool_reps(self):
        scores = np.array([[0.7, 0.8], [0.6, 0.9], [0.5, 0.7]], dtype=np.float32)
        pooled = _attention_pool_reps(scores)
        assert pooled.shape == (2,)
        # Pooled should be a weighted average between min and max
        assert pooled[0] >= 0.5 and pooled[0] <= 0.7
        assert pooled[1] >= 0.7 and pooled[1] <= 0.9

    def test_attention_pool_single_rep(self):
        scores = np.array([[0.6, 0.8]], dtype=np.float32)
        pooled = _attention_pool_reps(scores)
        np.testing.assert_allclose(pooled, [0.6, 0.8], atol=1e-5)


# ============================================================================
# Test: Feature Dimension Validation
# ============================================================================

class TestFeatureValidation:

    def test_valid_dimensions(self):
        feats = np.zeros((10, 50, 37))
        validate_feature_dimensions(feats, expected_dim=37)  # Should not raise

    def test_invalid_dimensions(self):
        feats = np.zeros((10, 50, 9))
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            validate_feature_dimensions(feats, expected_dim=37)


# ============================================================================
# Test: YAML Config Loading
# ============================================================================

class TestYAMLConfig:

    def test_config_loads(self):
        config = _get_rep_seg_config()
        assert isinstance(config, dict)
        # If the YAML file exists, it should have 'exercises' key
        config_path = PROJECT_ROOT / "config" / "rep_segmentation.yaml"
        if config_path.exists():
            assert "exercises" in config
            assert "defaults" in config
            assert "feature_mode" in config

    def test_config_contains_expected_exercises(self):
        config = _get_rep_seg_config()
        exercises = config.get("exercises", {})
        # Check a few known exercises
        for ex in ["lateral_raises", "hummer_curls", "weighted_squats", "deadlift"]:
            assert ex in exercises, f"Missing exercise '{ex}' in YAML config"

    def test_config_exercise_has_front_and_side(self):
        config = _get_rep_seg_config()
        exercises = config.get("exercises", {})
        for ex_name, views in exercises.items():
            assert "front" in views or "side" in views, (
                f"Exercise '{ex_name}' has neither 'front' nor 'side' view"
            )


# ============================================================================
# Test: Feature Name Constants
# ============================================================================

class TestFeatureConstants:

    def test_base_feature_names_count(self):
        assert len(ALL_FEATURE_NAMES) == 19

    def test_specialized_feature_names_count(self):
        assert len(SPECIALIZED_FEATURE_NAMES) == 18
        assert len(SIDE_SPECIALIZED_FEATURE_NAMES) == 18

    def test_extended_feature_names_count(self):
        assert len(ALL_EXTENDED_FEATURE_NAMES) == 37
        assert len(SIDE_ALL_EXTENDED_FEATURE_NAMES) == 37

    def test_no_duplicate_names(self):
        assert len(set(ALL_EXTENDED_FEATURE_NAMES)) == 37
        assert len(set(SIDE_ALL_EXTENDED_FEATURE_NAMES)) == 37


# ============================================================================
# Manual test runner (for environments without pytest)
# ============================================================================

def _run_manual_tests():
    """Run tests manually when pytest is not available."""
    print("=" * 70)
    print("Running quality assessment tests (manual mode)")
    print("=" * 70)

    test_classes = [
        TestLandmarkNormalization,
        TestFeatureExtraction,
        TestLegacyDeprecation,
        TestRepSegmentation,
        TestCNNModel,
        TestFeatureValidation,
        TestYAMLConfig,
        TestFeatureConstants,
    ]

    passed, failed = 0, 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✅ {cls.__name__}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
                failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        success = _run_manual_tests()
        sys.exit(0 if success else 1)
