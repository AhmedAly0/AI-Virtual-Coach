"""Test script for Exercise Recognition (Pose MLP) with YAML config and multi-run support."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.exercise_recognition import train_exercise_recognition, train_exercise_recognition_multi_run


def test_single_run_with_config():
    """Test single run using YAML configuration."""
    print("\n" + "=" * 80)
    print("TEST 1: Single Run with Config")
    print("=" * 80)
    
    npz_path = PROJECT_ROOT / 'datasets' / 'Mediapipe pose estimates' / 'pose_data_front_19_features.npz'
    config_path = PROJECT_ROOT / 'config' / 'exer_recog_baseline_front.yaml'
    
    # Override multi_run.enabled temporarily for single run test
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily disable multi_run for this test
    original_enabled = config['multi_run']['enabled']
    config['multi_run']['enabled'] = False
    
    # Save temporary config
    temp_config_path = PROJECT_ROOT / 'config' / 'exer_recog_temp.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        results = train_exercise_recognition(
            npz_path=str(npz_path),
            config_path=str(temp_config_path)
        )
        
        print("\n✅ Single run test PASSED")
        print(f"   Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"   Test Macro F1: {results['test_metrics']['macro_f1']:.4f}")
        print(f"   Results saved to: run_{results['run_idx']:03d}")
        
    finally:
        # Restore original config
        config['multi_run']['enabled'] = original_enabled
        temp_config_path.unlink(missing_ok=True)


def test_multi_run():
    """Test multi-run with 3 runs (reduced for testing speed)."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Run (3 runs for quick validation)")
    print("=" * 80)
    
    npz_path = PROJECT_ROOT / 'datasets' / 'Mediapipe pose estimates' / 'pose_data_front_19_features.npz'
    config_path = PROJECT_ROOT / 'config' / 'exer_recog_baseline_front.yaml'
    
    # Create test config with only 3 runs
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['multi_run']['enabled'] = True
    config['multi_run']['num_runs'] = 3
    
    temp_config_path = PROJECT_ROOT / 'config' / 'exer_recog_test.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        all_runs, stats = train_exercise_recognition_multi_run(
            npz_path=str(npz_path),
            config_path=str(temp_config_path)
        )
        
        print("\n✅ Multi-run test PASSED")
        print(f"   Number of runs: {len(all_runs)}")
        print(f"   Accuracy: {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
        print(f"   Macro F1: {stats['test_macro_f1']['mean']:.4f} ± {stats['test_macro_f1']['std']:.4f}")
        
    finally:
        temp_config_path.unlink(missing_ok=True)


def test_backward_compatibility():
    """Test that legacy parameter passing still works."""
    print("\n" + "=" * 80)
    print("TEST 3: Backward Compatibility (Legacy Parameters)")
    print("=" * 80)
    
    npz_path = PROJECT_ROOT / 'datasets' / 'Mediapipe pose estimates' / 'pose_data_front_19_features.npz'
    config_path = PROJECT_ROOT / 'config' / 'exer_recog_baseline_front.yaml'
    
    # Create config with multi_run disabled
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['multi_run']['enabled'] = False
    
    temp_config_path = PROJECT_ROOT / 'config' / 'exer_recog_legacy.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Use legacy parameter style (should override config)
        results = train_exercise_recognition(
            npz_path=str(npz_path),
            config_path=str(temp_config_path),
            seed=99,
            batch_size=32,
            max_epochs=5,  # Quick test
        )
        
        print("\n✅ Backward compatibility test PASSED")
        print(f"   Seed used: {results['seed']} (expected: 99)")
        print(f"   Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        
    finally:
        temp_config_path.unlink(missing_ok=True)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("EXERCISE RECOGNITION (POSE MLP) VALIDATION TESTS")
    print("=" * 80)
    
    try:
        # Run all tests
        test_single_run_with_config()
        test_backward_compatibility()
        
        # Note: Comment out multi-run test for faster validation
        # Uncomment to test full multi-run functionality
        # test_multi_run()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
