"""
Tests for Experiment 5 (Small CNN with GEI) - Data Splitting and Multi-Run Framework
"""

import pytest
import numpy as np
from src.data.data_loader import (
    split_by_subjects_three_way,
    split_by_subject_two_way,
    verify_subject_split_integrity
)


class TestSubjectSplitting:
    """Tests for subject-based stratified splitting functions."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with multiple subjects and exercises."""
        # 5 subjects, 3 exercises each (some exercises missing for some subjects)
        # Subject IDs: v1, v2, v3, v4, v5
        # Exercises: 0, 1, 2
        dataset = []
        
        # Subject v1: all 3 exercises
        for ex in range(3):
            for sample in range(2):  # 2 samples per exercise
                dataset.append({
                    'subject_id': 'v1',
                    'label': ex,
                    'data': np.random.rand(64, 64)
                })
        
        # Subject v2: exercises 0 and 1
        for ex in range(2):
            for sample in range(2):
                dataset.append({
                    'subject_id': 'v2',
                    'label': ex,
                    'data': np.random.rand(64, 64)
                })
        
        # Subject v3: all 3 exercises
        for ex in range(3):
            for sample in range(3):  # 3 samples per exercise
                dataset.append({
                    'subject_id': 'v3',
                    'label': ex,
                    'data': np.random.rand(64, 64)
                })
        
        # Subject v4: exercises 1 and 2
        for ex in range(1, 3):
            for sample in range(2):
                dataset.append({
                    'subject_id': 'v4',
                    'label': ex,
                    'data': np.random.rand(64, 64)
                })
        
        # Subject v5: all 3 exercises
        for ex in range(3):
            for sample in range(2):
                dataset.append({
                    'subject_id': 'v5',
                    'label': ex,
                    'data': np.random.rand(64, 64)
                })
        
        label_names = ['Exercise_0', 'Exercise_1', 'Exercise_2']
        
        return dataset, label_names
    
    def test_no_subject_overlap_stratified(self, sample_dataset):
        """Test that no subject appears in multiple splits (train/val/test)."""
        dataset, label_names = sample_dataset
        
        train_data, val_data, test_data = split_by_subjects_three_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        
        # Extract subject IDs from each split
        train_subjects = set(sample['subject_id'] for sample in train_data)
        val_subjects = set(sample['subject_id'] for sample in val_data)
        test_subjects = set(sample['subject_id'] for sample in test_data)
        
        # Check no overlap
        assert len(train_subjects & val_subjects) == 0, "Train and validation splits have overlapping subjects"
        assert len(train_subjects & test_subjects) == 0, "Train and test splits have overlapping subjects"
        assert len(val_subjects & test_subjects) == 0, "Validation and test splits have overlapping subjects"
        
        # Check all subjects are accounted for
        all_subjects_in_splits = train_subjects | val_subjects | test_subjects
        original_subjects = set(sample['subject_id'] for sample in dataset)
        assert all_subjects_in_splits == original_subjects, "Some subjects are missing from the splits"
    
    def test_all_samples_from_subject_in_same_split(self, sample_dataset):
        """Test that all samples from a subject (across all exercises) are in the same split."""
        dataset, label_names = sample_dataset
        
        train_data, val_data, test_data = split_by_subjects_three_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        
        # Build subject-to-split mapping
        subject_to_split = {}
        
        for sample in train_data:
            subject_id = sample['subject_id']
            if subject_id in subject_to_split:
                assert subject_to_split[subject_id] == 'train', f"Subject {subject_id} found in multiple splits"
            subject_to_split[subject_id] = 'train'
        
        for sample in val_data:
            subject_id = sample['subject_id']
            if subject_id in subject_to_split:
                assert subject_to_split[subject_id] == 'val', f"Subject {subject_id} found in multiple splits"
            subject_to_split[subject_id] = 'val'
        
        for sample in test_data:
            subject_id = sample['subject_id']
            if subject_id in subject_to_split:
                assert subject_to_split[subject_id] == 'test', f"Subject {subject_id} found in multiple splits"
            subject_to_split[subject_id] = 'test'
    
    def test_stratified_ensures_all_classes_in_test(self, sample_dataset):
        """Test that stratified splitting ensures all classes are in the test split."""
        dataset, label_names = sample_dataset
        
        train_data, val_data, test_data = split_by_subjects_three_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        
        # Check that test split has all classes
        test_labels = set(sample['label'] for sample in test_data)
        all_labels = set(range(len(label_names)))
        
        assert test_labels == all_labels, f"Test split missing classes: {all_labels - test_labels}"
        
        # Also check train and val
        train_labels = set(sample['label'] for sample in train_data)
        val_labels = set(sample['label'] for sample in val_data)
        
        assert train_labels == all_labels, f"Train split missing classes: {all_labels - train_labels}"
        assert val_labels == all_labels, f"Val split missing classes: {all_labels - val_labels}"
    
    def test_two_way_split_no_overlap(self, sample_dataset):
        """Test that two-way split also has no subject overlap."""
        dataset, label_names = sample_dataset
        
        train_data, test_data = split_by_subject_two_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.7,
            seed=42
        )
        
        # Extract subject IDs
        train_subjects = set(sample['subject_id'] for sample in train_data)
        test_subjects = set(sample['subject_id'] for sample in test_data)
        
        # Check no overlap
        assert len(train_subjects & test_subjects) == 0, "Train and test splits have overlapping subjects"
        
        # Check all subjects are accounted for
        all_subjects_in_splits = train_subjects | test_subjects
        original_subjects = set(sample['subject_id'] for sample in dataset)
        assert all_subjects_in_splits == original_subjects, "Some subjects are missing from the splits"
    
    def test_reproducibility_with_seed(self, sample_dataset):
        """Test that same seed produces same splits."""
        dataset, label_names = sample_dataset
        
        # First split
        train_data_1, val_data_1, test_data_1 = split_by_subjects_three_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=12345
        )
        
        # Second split with same seed
        train_data_2, val_data_2, test_data_2 = split_by_subjects_three_way(
            dataset=dataset,
            label_names=label_names,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=12345
        )
        
        # Extract subject IDs
        train_subjects_1 = set(sample['subject_id'] for sample in train_data_1)
        train_subjects_2 = set(sample['subject_id'] for sample in train_data_2)
        
        val_subjects_1 = set(sample['subject_id'] for sample in val_data_1)
        val_subjects_2 = set(sample['subject_id'] for sample in val_data_2)
        
        test_subjects_1 = set(sample['subject_id'] for sample in test_data_1)
        test_subjects_2 = set(sample['subject_id'] for sample in test_data_2)
        
        # Check that subjects are the same
        assert train_subjects_1 == train_subjects_2, "Train split not reproducible with same seed"
        assert val_subjects_1 == val_subjects_2, "Val split not reproducible with same seed"
        assert test_subjects_1 == test_subjects_2, "Test split not reproducible with same seed"


class TestVerificationUtility:
    """Tests for the verify_subject_split_integrity utility function."""
    
    @pytest.fixture
    def good_splits(self):
        """Create splits with no overlap."""
        train_data = [
            {'subject_id': 'v1', 'label': 0},
            {'subject_id': 'v1', 'label': 1},
            {'subject_id': 'v2', 'label': 0},
        ]
        val_data = [
            {'subject_id': 'v3', 'label': 0},
            {'subject_id': 'v3', 'label': 1},
        ]
        test_data = [
            {'subject_id': 'v4', 'label': 0},
            {'subject_id': 'v5', 'label': 1},
        ]
        return train_data, val_data, test_data
    
    @pytest.fixture
    def bad_splits_with_overlap(self):
        """Create splits with subject overlap."""
        train_data = [
            {'subject_id': 'v1', 'label': 0},
            {'subject_id': 'v2', 'label': 0},
        ]
        val_data = [
            {'subject_id': 'v2', 'label': 1},  # v2 also in train!
            {'subject_id': 'v3', 'label': 0},
        ]
        test_data = [
            {'subject_id': 'v4', 'label': 0},
        ]
        return train_data, val_data, test_data
    
    def test_detects_overlap(self, bad_splits_with_overlap):
        """Test that verification utility detects subject overlap."""
        train_data, val_data, test_data = bad_splits_with_overlap
        
        with pytest.raises(AssertionError, match="Found .* overlapping subjects"):
            verify_subject_split_integrity(train_data, val_data, test_data)
    
    def test_no_false_positives(self, good_splits):
        """Test that verification utility doesn't flag good splits."""
        train_data, val_data, test_data = good_splits
        
        # Should not raise any assertion errors
        verify_subject_split_integrity(train_data, val_data, test_data)


class TestMultiRunResults:
    """Tests for multi-run experiment framework."""
    
    @pytest.fixture
    def mock_multi_run_results(self):
        """Create mock results from multiple runs."""
        results = []
        for run_idx in range(5):
            run_result = {
                'run_idx': run_idx,
                'seed': 42 + run_idx,
                'test_accuracy': 0.85 + np.random.rand() * 0.1,
                'test_macro_f1': 0.83 + np.random.rand() * 0.1,
                'test_confusion_matrix': np.random.randint(0, 50, size=(15, 15)).tolist(),
                'per_class_f1': {str(i): 0.8 + np.random.rand() * 0.15 for i in range(15)}
            }
            results.append(run_result)
        return results
    
    def test_aggregation_computes_mean_std(self, mock_multi_run_results):
        """Test that aggregation correctly computes mean and std across runs."""
        # Extract test accuracies
        test_accuracies = [run['test_accuracy'] for run in mock_multi_run_results]
        
        # Compute mean and std
        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies, ddof=1)  # Sample std
        
        # Check reasonable values
        assert 0.0 <= mean_acc <= 1.0, "Mean accuracy out of valid range"
        assert std_acc >= 0.0, "Std should be non-negative"
        assert std_acc < 1.0, "Std seems too large for accuracy values"
    
    def test_all_runs_have_required_fields(self, mock_multi_run_results):
        """Test that all runs contain the required fields."""
        required_fields = ['run_idx', 'seed', 'test_accuracy', 'test_macro_f1', 
                          'test_confusion_matrix', 'per_class_f1']
        
        for run_result in mock_multi_run_results:
            for field in required_fields:
                assert field in run_result, f"Run {run_result.get('run_idx', '?')} missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
