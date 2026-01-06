"""
Data loading utilities for GEI (Gait Energy Image) datasets.

This module handles loading GEI images from folder structures and splitting
datasets by subject IDs to prevent data leakage.
"""

import os
import re
import cv2
import random
import numpy as np
import logging
from collections import Counter
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Only add handler if no root handlers exist (notebook provides central logging)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(_handler)
logger.propagate = True  # Let messages propagate to root handler


def _load_pose_npz(npz_path: str, view_label: str) -> Tuple[List[Tuple[str, np.ndarray, str, str]], Dict]:
    """Load pose static features from a NPZ file and tag with view metadata.

    Args:
        npz_path (str): Path to the pose NPZ file to load.
        view_label (str): Human-readable view tag (e.g., "front" or "side").

    Returns:
        Tuple[List[Tuple[str, np.ndarray, str, str]], Dict]: Parsed dataset
        as (exercise_name, static_features, subject_id, view) tuples and a summary dict
        with keys: count, unique_subjects, unique_classes, view, angle_names.
    
    Raises:
        KeyError: If NPZ file is missing required 'angle_names' field.
    """

    data = np.load(npz_path, allow_pickle=True)
    static_features = data['X_static']  # Shape: (num_reps, 45)
    exercise_names = data['exercise_names']
    subject_ids = data['subject_ids']  # Integer array
    
    # Read angle_names from NPZ (required field)
    if 'angle_names' not in data:
        raise KeyError(
            f"NPZ file at {npz_path} is missing required 'angle_names' field. "
            "Please regenerate the NPZ file with preprocessing script."
        )
    angle_names = [str(name) for name in data['angle_names']]

    dataset: List[Tuple[str, np.ndarray, str, str]] = []
    for features, exercise_name, subject_id in zip(static_features, exercise_names, subject_ids):
        # Convert integer subject_id to normalized string format
        subject = _normalize_subject_id(str(int(subject_id)))
        features_array = np.asarray(features, dtype=np.float32)
        dataset.append((exercise_name, features_array, subject, view_label))

    summary = {
        'count': len(dataset),
        'unique_subjects': len(set(item[2] for item in dataset)),
        'unique_classes': len(set(item[0] for item in dataset)),
        'view': view_label,
        'angle_names': angle_names,
    }
    return dataset, summary


def load_pose_data(npz_path: str) -> Tuple[List[Tuple[str, np.ndarray, str, str]], Dict]:
    """Load pose static features from a single NPZ file.

    Args:
        npz_path (str): Path to pose static NPZ file (front or side view).

    Returns:
        Tuple containing:
        - dataset: List of (exercise_name, static_features, subject_id, view) tuples
        - summary: Dict with keys: count, unique_subjects, unique_classes, view, angle_names

    Raises:
        FileNotFoundError: If NPZ file doesn't exist.
        KeyError: If NPZ file is missing required 'angle_names' field.
    """

    # Infer view from filename
    view_label = 'front' if 'front' in str(npz_path).lower() else 'side'
    
    dataset, summary = _load_pose_npz(npz_path, view_label=view_label)

    logger.info(
        "[load_pose_data] Loaded %s samples (%s subjects, %s classes) from %s view",
        summary['count'],
        summary['unique_subjects'],
        summary['unique_classes'],
        view_label,
    )
    return dataset, summary


def load_pose_temporal_data(npz_path: str) -> Tuple[List[Tuple[str, np.ndarray, str, str]], Dict]:
    """Load pose temporal features from a single NPZ file.

    Args:
        npz_path (str): Path to pose temporal NPZ file (front or side view).

    Returns:
        Tuple containing:
        - dataset: List of (exercise_name, temporal_features, subject_id, view) tuples
          where temporal_features has shape (T_fixed, num_angles), e.g., (50, 9)
        - summary: Dict with keys: count, unique_subjects, unique_classes, view, 
          angle_names, temporal_shape

    Raises:
        FileNotFoundError: If NPZ file doesn't exist.
        KeyError: If NPZ file is missing required fields.
    """

    # Infer view from filename
    view_label = 'front' if 'front' in str(npz_path).lower() else 'side'
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Load temporal features (shape: (num_reps, T_fixed, num_angles))
    if 'X_temporal' not in data:
        raise KeyError(
            f"NPZ file at {npz_path} is missing required 'X_temporal' field. "
            "Please use a temporal pose NPZ file."
        )
    
    temporal_features = data['X_temporal']  # Shape: (num_reps, 50, 9)
    exercise_names = data['exercise_names']
    subject_ids = data['subject_ids']  # Integer array
    
    # Read angle_names from NPZ (required field)
    if 'angle_names' not in data:
        raise KeyError(
            f"NPZ file at {npz_path} is missing required 'angle_names' field. "
            "Please regenerate the NPZ file with preprocessing script."
        )
    angle_names = [str(name) for name in data['angle_names']]

    dataset: List[Tuple[str, np.ndarray, str, str]] = []
    for features, exercise_name, subject_id in zip(temporal_features, exercise_names, subject_ids):
        # Convert integer subject_id to normalized string format
        subject = _normalize_subject_id(str(int(subject_id)))
        features_array = np.asarray(features, dtype=np.float32)  # Shape: (T_fixed, num_angles)
        dataset.append((exercise_name, features_array, subject, view_label))

    temporal_shape = temporal_features.shape[1:]  # (T_fixed, num_angles), e.g., (50, 9)
    
    summary = {
        'count': len(dataset),
        'unique_subjects': len(set(item[2] for item in dataset)),
        'unique_classes': len(set(item[0] for item in dataset)),
        'view': view_label,
        'angle_names': angle_names,
        'temporal_shape': temporal_shape,
    }

    logger.info(
        "[load_pose_temporal_data] Loaded %s samples (%s subjects, %s classes) from %s view, temporal shape: %s",
        summary['count'],
        summary['unique_subjects'],
        summary['unique_classes'],
        view_label,
        temporal_shape,
    )
    return dataset, summary


def _normalize_subject_id(subject_id: str) -> str:
    """Canonicalize folder-level subject identifiers for reliable splits.

    Args:
        subject_id (str): Raw folder name pulled from the dataset tree.

    Returns:
        str: Standardized identifier (e.g., ``volunteer_007``) used across
            loading/splitting utilities. Falls back to ``volunteer_unknown``
            when no characters are available.
    """

    if subject_id is None:
        return 'volunteer_unknown'

    cleaned = subject_id.strip()
    if not cleaned:
        return 'volunteer_unknown'

    match = re.search(r"(\d+)", cleaned)
    if match:
        number = int(match.group(1))
        return f"volunteer_{number:03d}"

    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip('_')
    return slug or 'volunteer_unknown'


def load_data(base_folder: str) -> List[Tuple[str, np.ndarray, str]]:
    """
    Load GEI images from nested folder structure.
    
    Expected structure:
        base_folder/
        ├── exercise1/
        │   ├── subject1/
        │   │   ├── image1.png
        │   │   └── image2.png
        │   └── subject2/
        │       └── ...
        └── exercise2/
            └── ...
    
    Args:
        base_folder (str): Root directory containing exercise folders
        
    Returns:
        List[Tuple]: List of (exercise_name, image_array, subject_id) tuples with
            subject IDs normalized via `_normalize_subject_id`.
        
    Raises:
        ValueError: If no valid images are found in the base folder
    """
    dataset = []
    
    # Level 1: Exercise folders
    for exercise_name in os.listdir(base_folder):
        exercise_path = os.path.join(base_folder, exercise_name)
        
        if not os.path.isdir(exercise_path):
            continue
        
        # Level 2: Subject folders
        for subject_id in os.listdir(exercise_path):
            subject_path = os.path.join(exercise_path, subject_id)
            
            if not os.path.isdir(subject_path):
                continue
            
            # Level 3: Image files
            for image_filename in os.listdir(subject_path):
                if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                image_path = os.path.join(subject_path, image_filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    normalized_subject = _normalize_subject_id(subject_id)
                    dataset.append((exercise_name, img, normalized_subject))
    
    if not dataset:
        raise ValueError(f"No valid images found in {base_folder}")
    
    logger.info(f"[load_data] Loaded {len(dataset)} samples from '{base_folder}'.")
    return dataset


def get_subjects_identities(dataset: List[Tuple[str, np.ndarray, str]]) -> List[str]:
    """
    Extract unique subject IDs from the dataset.
    
    Args:
        dataset (List[Tuple]): List of (exercise_name, image, subject_id) tuples
        
    Returns:
        List[str]: Sorted list of unique normalized subject IDs
        
    Raises:
        ValueError: If dataset is empty
    """
    if not dataset:
        logger.warning("Empty dataset provided to get_subjects_identities")
        return []
    
    identities = sorted(set(item[2] for item in dataset))
    return identities


def split_by_subject_two_way(
    dataset: List[Tuple[str, np.ndarray, str]],
    split_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]]]:
    """Unified 2-way subject-wise split with optional stratification.
    
    This function ensures that all samples from the same subject are either in the
    first set or the second set, never split between both. When stratified=True,
    attempts to ensure all exercise classes are represented in both splits.
    
    Args:
        dataset (List[Tuple]): List of (exercise_name, image, subject_id) tuples
        split_ratio (float): Fraction of subjects for the second split (0 < ratio < 1)
        seed (int): Random seed for reproducible splitting
        stratified (bool): If True, ensures each exercise class has ≥1 subject in both splits
        
    Returns:
        Tuple[List, List]: (larger_set, smaller_set) where smaller_set has ~split_ratio subjects
        
    Raises:
        ValueError: If split_ratio is not between 0 and 1, or if dataset is empty
        
    Example:
        # Simple random split (backward compatible)
        train, test = split_by_subject_two_way(dataset, split_ratio=0.3, seed=42)
        
        # Stratified split (ensures all 15 exercises in both train and test)
        train, test = split_by_subject_two_way(dataset, split_ratio=0.3, seed=42, stratified=True)
    """
    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    
    if not dataset:
        raise ValueError("Dataset is empty")
    
    subjects = get_subjects_identities(dataset)
    rng = random.Random(seed)
    
    if stratified:
        # Build subject→exercises mapping to handle multi-exercise subjects
        subject_to_exercises: Dict[str, set] = {}
        for exercise, _, subject, *_ in dataset:
            if subject not in subject_to_exercises:
                subject_to_exercises[subject] = set()
            subject_to_exercises[subject].add(exercise)
        
        # Group exercises by their subject lists (but track globally assigned subjects)
        exercise_to_subjects: Dict[str, List[str]] = {}
        for subject, exercises in subject_to_exercises.items():
            for exercise in exercises:
                if exercise not in exercise_to_subjects:
                    exercise_to_subjects[exercise] = []
                if subject not in exercise_to_subjects[exercise]:
                    exercise_to_subjects[exercise].append(subject)
        
        # Shuffle subject lists per exercise
        for exercise in exercise_to_subjects:
            rng.shuffle(exercise_to_subjects[exercise])
        
        # Global tracking to prevent subject duplication across splits
        set1_subjects = set()
        set2_subjects = set()
        assigned_subjects = set()
        
        # First pass: assign subjects to ensure all classes represented in both splits
        for exercise, class_subjects in exercise_to_subjects.items():
            # Only consider unassigned subjects for this class
            available = [s for s in class_subjects if s not in assigned_subjects]
            
            if not available:
                continue
            
            n_available = len(available)
            n_set2 = max(1, int(n_available * split_ratio))
            n_set1 = n_available - n_set2
            
            # Ensure at least 1 subject per split if possible
            if n_available >= 2:
                n_set2 = max(1, n_set2)
                n_set1 = max(1, n_available - n_set2)
            else:
                # Only 1 subject available - assign to larger split
                n_set1 = 1
                n_set2 = 0
                logger.warning(f"Class '{exercise}' has only 1 unassigned subject, assigned to larger split")
            
            # Assign subjects
            new_set2 = available[:n_set2]
            new_set1 = available[n_set2:n_set2 + n_set1]
            
            set2_subjects.update(new_set2)
            set1_subjects.update(new_set1)
            assigned_subjects.update(new_set2 + new_set1)
        
        # Second pass: assign any remaining unassigned subjects to larger split
        unassigned = set(subjects) - assigned_subjects
        if unassigned:
            logger.info(f"Assigning {len(unassigned)} remaining subjects to larger split")
            set1_subjects.update(unassigned)
        
        # Verify no overlap
        overlap = set1_subjects & set2_subjects
        if overlap:
            raise RuntimeError(f"BUG: Subject overlap detected in stratified split: {overlap}")
        
        # Verify coverage
        missing_from_set1 = [ex for ex in exercise_to_subjects if not any(s in set1_subjects for s in exercise_to_subjects[ex])]
        missing_from_set2 = [ex for ex in exercise_to_subjects if not any(s in set2_subjects for s in exercise_to_subjects[ex])]
        
        if missing_from_set2:
            logger.warning(f"Smaller split missing classes: {missing_from_set2}")
    else:
        # Simple random split (backward compatible)
        rng.shuffle(subjects)
        n_set2 = max(1, int(len(subjects) * split_ratio))
        set2_subjects = set(subjects[:n_set2])
        set1_subjects = set(subjects[n_set2:])
    
    # Partition dataset by subject membership
    set1_dataset = [item for item in dataset if item[2] in set1_subjects]
    set2_dataset = [item for item in dataset if item[2] in set2_subjects]
    
    logger.info(
        f"Split: {len(set1_dataset)} samples ({len(set1_subjects)} subjects), "
        f"{len(set2_dataset)} samples ({len(set2_subjects)} subjects)"
    )
    
    return set1_dataset, set2_dataset


def split_by_subjects_three_way(
    dataset: List[Tuple[str, np.ndarray, str]], 
    val_ratio: float = 0.15,
    test_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """Split dataset into train/val/test by subject IDs with optional stratification.
    
    Ensures that all samples from a subject go to the same group (train/val/test),
    preventing data leakage. When stratified=True, ensures all exercise classes are
    represented in ALL three splits (train, val, and test).
    
    Args:
        dataset (List[Tuple]): List of (exercise_name, image, subject_id) tuples
        val_ratio (float): Fraction of subjects for validation (default: 0.15)
        test_ratio (float): Fraction of subjects for test (default: 0.3)
        seed (int): Random seed for reproducible splitting (default: None)
        stratified (bool): If True, ensures each exercise class has ≥1 subject in ALL splits
        
    Returns:
        Tuple[List, List, List]: (train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If val_ratio + test_ratio >= 1, or if dataset is empty
        RuntimeError: If subject overlap is detected (indicates a bug)
        
    Example:
        # Stratified split ensuring all 15 exercises in train/val/test
        train, val, test = split_by_subjects_three_way(
            dataset, val_ratio=0.15, test_ratio=0.3, seed=42, stratified=True
        )
    """
    if not 0 < val_ratio + test_ratio < 1:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")
    
    if not dataset:
        raise ValueError("Dataset is empty")
    
    subjects = get_subjects_identities(dataset)
    rng = random.Random(seed)
    
    if stratified:
        # Build subject→exercises mapping to handle multi-exercise subjects
        subject_to_exercises: Dict[str, set] = {}
        for exercise, _, subject, *_ in dataset:
            if subject not in subject_to_exercises:
                subject_to_exercises[subject] = set()
            subject_to_exercises[subject].add(exercise)
        
        # Group exercises by their subject lists (but track globally assigned subjects)
        exercise_to_subjects: Dict[str, List[str]] = {}
        for subject, exercises in subject_to_exercises.items():
            for exercise in exercises:
                if exercise not in exercise_to_subjects:
                    exercise_to_subjects[exercise] = []
                if subject not in exercise_to_subjects[exercise]:
                    exercise_to_subjects[exercise].append(subject)
        
        # Shuffle subject lists per exercise
        for exercise in exercise_to_subjects:
            rng.shuffle(exercise_to_subjects[exercise])
        
        # Global tracking to prevent subject duplication across splits
        train_subjects = set()
        val_subjects = set()
        test_subjects = set()
        assigned_subjects = set()
        
        # Priority: ensure all classes in ALL THREE SPLITS (train, val, and test)
        for exercise, class_subjects in exercise_to_subjects.items():
            # Only consider unassigned subjects for this class
            available = [s for s in class_subjects if s not in assigned_subjects]
            
            if not available:
                continue
            
            n_available = len(available)
            n_test = max(1, int(n_available * test_ratio))
            n_val = max(1, int(n_available * val_ratio))  # Val must have ≥1 for each class
            n_train = n_available - n_test - n_val
            
            # Adjust if totals don't add up (edge case with small class sizes)
            if n_train + n_val + n_test > n_available:
                if n_available >= 3:
                    # Ensure all three splits get at least 1 subject
                    n_test = 1
                    n_val = 1
                    n_train = n_available - 2
                elif n_available == 2:
                    # Can't satisfy all three - prioritize train and test
                    n_train = 1
                    n_test = 1
                    n_val = 0
                    logger.warning(
                        f"Class '{exercise}' has only 2 unassigned subjects, "
                        f"cannot ensure presence in all 3 splits (missing from val)"
                    )
                else:
                    # Only 1 subject - can only go to one split
                    n_train = 1
                    n_test = 0
                    n_val = 0
                    logger.warning(
                        f"Class '{exercise}' has only 1 unassigned subject, assigned to train only"
                    )
            
            # Ensure all splits have at least 1 subject when possible
            if n_train < 1 and n_available >= 3:
                n_train = 1
                n_val = max(1, n_val)
                n_test = max(1, n_available - n_train - n_val)
            elif n_val < 1 and n_available >= 3:
                n_val = 1
                n_train = max(1, n_train)
                n_test = max(1, n_available - n_train - n_val)
            elif n_test < 1 and n_available >= 2:
                n_test = 1
                n_train = max(1, n_available - n_test - n_val)
            
            # Assign subjects
            new_test = available[:n_test]
            new_val = available[n_test:n_test + n_val]
            new_train = available[n_test + n_val:n_test + n_val + n_train]
            
            test_subjects.update(new_test)
            val_subjects.update(new_val)
            train_subjects.update(new_train)
            assigned_subjects.update(new_test + new_val + new_train)
        
        # Assign any remaining unassigned subjects to train
        unassigned = set(subjects) - assigned_subjects
        if unassigned:
            logger.info(f"Assigning {len(unassigned)} remaining subjects to train split")
            train_subjects.update(unassigned)
        
        # CRITICAL: Verify no overlap between splits
        train_val_overlap = train_subjects & val_subjects
        train_test_overlap = train_subjects & test_subjects
        val_test_overlap = val_subjects & test_subjects
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            error_msg = "BUG: Subject overlap detected in stratified split!\n"
            if train_val_overlap:
                error_msg += f"  Train-Val overlap: {train_val_overlap}\n"
            if train_test_overlap:
                error_msg += f"  Train-Test overlap: {train_test_overlap}\n"
            if val_test_overlap:
                error_msg += f"  Val-Test overlap: {val_test_overlap}\n"
            raise RuntimeError(error_msg)
        
        # Verify all subjects accounted for
        total_assigned = train_subjects | val_subjects | test_subjects
        if len(total_assigned) != len(subjects):
            logger.warning(
                f"Subject accounting mismatch: {len(total_assigned)} assigned vs {len(subjects)} total"
            )
        
        # Verify coverage (all splits should have all classes)
        missing_from_train = [ex for ex in exercise_to_subjects if not any(s in train_subjects for s in exercise_to_subjects[ex])]
        missing_from_test = [ex for ex in exercise_to_subjects if not any(s in test_subjects for s in exercise_to_subjects[ex])]
        missing_from_val = [ex for ex in exercise_to_subjects if not any(s in val_subjects for s in exercise_to_subjects[ex])]
        
        if missing_from_train:
            logger.error(f"Train split missing classes: {missing_from_train}")
        if missing_from_test:
            logger.error(f"Test split missing classes: {missing_from_test}")
        if missing_from_val:
            logger.warning(f"Val split missing classes: {missing_from_val}")
    else:
        # Simple random split (backward compatible)
        rng.shuffle(subjects)
        n_subjects = len(subjects)
        n_test = max(1, int(n_subjects * test_ratio))
        n_val = max(1, int(n_subjects * val_ratio))
        
        test_subjects = set(subjects[:n_test])
        val_subjects = set(subjects[n_test:n_test + n_val])
        train_subjects = set(subjects[n_test + n_val:])
    
    # Assign samples based on subject group
    train_dataset = [item for item in dataset if item[2] in train_subjects]
    val_dataset = [item for item in dataset if item[2] in val_subjects]
    test_dataset = [item for item in dataset if item[2] in test_subjects]
    
    logger.info(
        f"Subject split: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects"
    )
    
    return train_dataset, val_dataset, test_dataset


def build_subject_folds(
    dataset: List[Tuple[str, np.ndarray, str]],
    num_folds: int = 5,
    seed: int = None,
    stratified: bool = True
) -> List[List[Tuple[str, np.ndarray, str]]]:
    """Create subject-wise folds with optional stratification across labels.

    Args:
        dataset (List[Tuple[str, np.ndarray, str]]): Collection of (label, image,
            subject_id) tuples.
        num_folds (int): Desired number of subject-disjoint folds (>=2).
        seed (int, optional): Seed controlling both subject ordering and
            deterministic tie-breakers.
        stratified (bool): When ``True`` attempts to balance label counts across
            folds using greedy assignment; otherwise balances purely by subject
            volume.

    Returns:
        List[List[Tuple[str, np.ndarray, str]]]: List of folds, each containing
        subject-exclusive samples ready for cross-validation.

    Raises:
        ValueError: If the dataset is empty, ``num_folds`` is invalid, or there are
        insufficient unique subjects to populate every fold.
    """

    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")

    if not dataset:
        raise ValueError("Dataset cannot be empty")

    subjects: Dict[str, Dict] = {}
    for label, image, subject in dataset:
        subject_entry = subjects.setdefault(subject, {'samples': [], 'label_counts': Counter()})
        subject_entry['samples'].append((label, image, subject))
        subject_entry['label_counts'][label] += 1

    subject_ids = list(subjects.keys())
    if len(subject_ids) < num_folds:
        raise ValueError(
            f"Not enough unique subjects ({len(subject_ids)}) to populate {num_folds} folds."
        )

    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    # Order by sample count (desc) while keeping randomness for ties
    subject_ids.sort(key=lambda sid: (-sum(subjects[sid]['label_counts'].values()), rng.random()))

    fold_subject_members: List[List[str]] = [[] for _ in range(num_folds)]
    fold_label_counts: List[Counter] = [Counter() for _ in range(num_folds)]

    for subject_id in subject_ids:
        subject_info = subjects[subject_id]

        if stratified:
            def score(fold_idx: int) -> Tuple[int, int, float]:
                label_overlap = sum(
                    fold_label_counts[fold_idx][label]
                    for label in subject_info['label_counts']
                )
                return (
                    label_overlap,
                    len(fold_subject_members[fold_idx]),
                    rng.random(),
                )

            target_fold = min(range(num_folds), key=score)
        else:
            target_fold = min(
                range(num_folds),
                key=lambda idx: (len(fold_subject_members[idx]), rng.random())
            )

        fold_subject_members[target_fold].append(subject_id)
        fold_label_counts[target_fold].update(subject_info['label_counts'])

    empty_indices = [idx for idx, members in enumerate(fold_subject_members) if not members]
    if empty_indices:
        logger.warning(
            "Detected %s empty folds after initial assignment; rebalancing subjects.",
            len(empty_indices),
        )
        for empty_idx in empty_indices:
            donor_candidates = sorted(
                [idx for idx, members in enumerate(fold_subject_members) if members],
                key=lambda idx: len(fold_subject_members[idx]),
                reverse=True,
            )
            moved = False
            for donor_idx in donor_candidates:
                if len(fold_subject_members[donor_idx]) <= 1:
                    continue
                subject_to_move = fold_subject_members[donor_idx].pop()
                fold_subject_members[empty_idx].append(subject_to_move)
                moved = True
                break
            if not moved:
                raise ValueError(
                    "Unable to rebalance folds without leaving some empty. Consider "
                    "reducing num_folds or collecting more subjects."
                )

    folds: List[List[Tuple[str, np.ndarray, str]]] = []
    for member_ids in fold_subject_members:
        fold_samples: List[Tuple[str, np.ndarray, str]] = []
        for subject_id in member_ids:
            fold_samples.extend(subjects[subject_id]['samples'])
        folds.append(fold_samples)

    for idx, fold in enumerate(folds):
        logger.info(
            "Fold %s: %s samples, %s subjects",
            idx,
            len(fold),
            len({sample[2] for sample in fold})
        )

    return folds


def verify_subject_split_integrity(
    train_samples: List[Tuple[str, np.ndarray, str]],
    val_samples: List[Tuple[str, np.ndarray, str]] = None,
    test_samples: List[Tuple[str, np.ndarray, str]] = None,
    *,
    verbose: bool = True
) -> Dict:
    """Verify that no subject appears in multiple splits (data leakage check).
    
    Args:
        train_samples: Training samples [(exercise, image, subject_id), ...]
        val_samples: Validation samples (optional for 2-way splits)
        test_samples: Test samples
        verbose: If True, log detailed diagnostic information
        
    Returns:
        Dict containing verification results and overlap details
        
    Raises:
        RuntimeError: If subject overlap is detected between any splits
    """
    # Extract unique subjects from each split
    train_subjects = set(sample[2] for sample in train_samples)
    val_subjects = set(sample[2] for sample in val_samples) if val_samples else set()
    test_subjects = set(sample[2] for sample in test_samples) if test_samples else set()
    
    # Check for overlaps
    train_val_overlap = train_subjects & val_subjects if val_subjects else set()
    train_test_overlap = train_subjects & test_subjects if test_subjects else set()
    val_test_overlap = val_subjects & test_subjects if val_subjects and test_subjects else set()
    
    has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)
    
    # Check class distribution per split
    train_classes = set(sample[0] for sample in train_samples)
    val_classes = set(sample[0] for sample in val_samples) if val_samples else set()
    test_classes = set(sample[0] for sample in test_samples) if test_samples else set()
    all_classes = train_classes | val_classes | test_classes
    
    results = {
        'has_subject_overlap': has_overlap,
        'train_val_overlap': sorted(train_val_overlap),
        'train_test_overlap': sorted(train_test_overlap),
        'val_test_overlap': sorted(val_test_overlap),
        'total_unique_subjects': len(train_subjects | val_subjects | test_subjects),
        'train_subjects_count': len(train_subjects),
        'val_subjects_count': len(val_subjects),
        'test_subjects_count': len(test_subjects),
        'total_classes': len(all_classes),
        'train_classes': len(train_classes),
        'val_classes': len(val_classes),
        'test_classes': len(test_classes),
        'train_missing_classes': sorted(all_classes - train_classes),
        'val_missing_classes': sorted(all_classes - val_classes) if val_classes else [],
        'test_missing_classes': sorted(all_classes - test_classes) if test_classes else [],
    }
    
    if verbose:
        logger.info("=" * 70)
        logger.info("SUBJECT SPLIT INTEGRITY VERIFICATION")
        logger.info("=" * 70)
        logger.info(
            f"Subject distribution: Train={results['train_subjects_count']}, "
            f"Val={results['val_subjects_count']}, Test={results['test_subjects_count']}, "
            f"Total={results['total_unique_subjects']}"
        )
        
        if has_overlap:
            error_msg = "❌ DATA LEAKAGE DETECTED! Subject overlap found:\n"
            if train_val_overlap:
                error_msg += f"  Train-Val overlap: {len(train_val_overlap)} subjects\n"
            if train_test_overlap:
                error_msg += f"  Train-Test overlap: {len(train_test_overlap)} subjects\n"
            if val_test_overlap:
                error_msg += f"  Val-Test overlap: {len(val_test_overlap)} subjects\n"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.info("✅ No subject overlap - splits are valid!")
        
        logger.info(
            f"Class distribution: Total={results['total_classes']}, "
            f"Train={results['train_classes']}, Val={results['val_classes']}, "
            f"Test={results['test_classes']}"
        )
        
        if results['train_missing_classes']:
            logger.error(f"Train missing classes: {results['train_missing_classes']}")
        if results['test_missing_classes']:
            logger.error(f"Test missing classes: {results['test_missing_classes']}")
        if results['val_missing_classes']:
            logger.warning(f"Val missing classes: {results['val_missing_classes']}")
        
        logger.info("=" * 70)
    
    return results


def load_front_side_geis(
    front_base_folder: str,
    side_base_folder: str,
    *,
    seed: int = 42,
    shuffle: bool = True
) -> Tuple[List[Tuple[str, np.ndarray, str]], Dict[str, int]]:
    """Load and fuse front/side GEIs, mirroring the notebook logic.

    Args:
        front_base_folder (str): Path to the front-view GEI directory structure.
        side_base_folder (str): Path to the side-view GEI directory structure.
        seed (int): RNG seed used when shuffling the combined dataset.
        shuffle (bool): Whether to shuffle the concatenated dataset before return.

    Returns:
        Tuple[List[Tuple[str, np.ndarray, str]], Dict[str, int]]: The merged dataset
        and a summary dictionary containing the individual/front/side counts.
    """

    front_dataset = load_data(front_base_folder)
    side_dataset = load_data(side_base_folder)
    combined_dataset = front_dataset + side_dataset

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(combined_dataset)

    summary = {
        'front_count': len(front_dataset),
        'side_count': len(side_dataset),
        'total_count': len(combined_dataset),
    }

    logger.info(
        "Combined dataset: %s total samples (front=%s, side=%s)",
        summary['total_count'],
        summary['front_count'],
        summary['side_count'],
    )

    return combined_dataset, summary
