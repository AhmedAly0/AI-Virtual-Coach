"""
TensorFlow dataset building utilities for GEI training.

This module creates train/test splits and builds tf.data.Dataset pipelines
with augmentation and batching.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .data_loader import split_by_subject_two_way, split_by_subjects_three_way
from .preprocessing import (
    prep_tensors_with_preprocess,
    to_int,
    _resize_and_stack_to_rgb,
    _resize_to_grayscale,
)
from .augmentation import data_augmentater


def _pose_sequence_to_features(seq: 'np.ndarray') -> 'np.ndarray':
    """Convert a variable-length pose sequence into a fixed-size feature vector.

    Args:
        seq (np.ndarray): Pose sequence shaped (timesteps, num_joints) or (timesteps,).

    Returns:
        np.ndarray: Concatenated mean and std features per joint, with NaNs replaced by zeros.
    """

    seq_arr = np.asarray(seq, dtype=np.float32)
    if seq_arr.ndim == 1:
        seq_arr = seq_arr[:, None]

    mean_feats = np.nanmean(seq_arr, axis=0)
    std_feats = np.nanstd(seq_arr, axis=0)
    feats = np.concatenate([mean_feats, std_feats]).astype(np.float32)
    return np.nan_to_num(feats)


def _pose_samples_to_arrays(samples: List[Tuple[str, 'np.ndarray', str, str]], label_to_int: Dict[str, int]):
    """Vectorize pose samples into feature matrix and label vector.

    Args:
        samples (List[Tuple]): (exercise_name, static_features, subject_id, view) tuples.
        label_to_int (Dict[str, int]): Mapping from class label to integer id.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix (N, F) and label vector (N,).
    """

    # Data is already static features (45-dim), no aggregation needed
    X = np.stack([sample[1] for sample in samples]).astype(np.float32)
    y = np.array([label_to_int[sample[0]] for sample in samples], dtype=np.int32)
    return X, y

logger = logging.getLogger(__name__)


def make_split(
    dataset: List[Tuple[str, 'np.ndarray', str]], 
    test_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List[Tuple], List[Tuple], Dict[str, int]]:
    """
    Split dataset into train/test by subjects and create label mapping.
    
    Args:
        dataset (List[Tuple]): Full dataset of (exercise_name, image, subject_id)
        test_ratio (float): Fraction of subjects to use for testing
        seed (int): Random seed for reproducible splitting
        stratified (bool): If True, ensures all exercise classes in both train/test
        
    Returns:
        Tuple: (train_samples, test_samples, label_to_int_mapping)
    """
    train_samples, test_samples = split_by_subject_two_way(
        dataset, split_ratio=test_ratio, seed=seed, stratified=stratified
    )
    
    # Create label mapping from all labels in dataset
    all_labels = [item[0] for item in dataset]
    label_to_int = to_int(all_labels)
    
    logger.info(f"Dataset split: {len(train_samples)} train, {len(test_samples)} test")
    logger.info(f"Number of classes: {len(label_to_int)}")
    
    return train_samples, test_samples, label_to_int


def build_datasets(
    train_samples: List[Tuple],
    test_samples: List[Tuple],
    label_to_int: Dict[str, int],
    batch_size: int = 32,
    img_size: int = 224,
    preprocess_fn = None,
    augment_config: Optional[Dict] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray']:
    """
    Build TensorFlow datasets with augmentation and batching.
    
    Creates optimized tf.data.Dataset pipelines for training and testing:
    - Training: shuffled, augmented, batched, prefetched
    - Testing: batched, prefetched (no augmentation)
    
    Args:
        train_samples (List[Tuple]): Training samples
        test_samples (List[Tuple]): Test samples
        label_to_int (Dict): Label to integer mapping
        batch_size (int): Batch size for training and testing
        img_size (int): Target image size
        preprocess_fn (callable, optional): Backbone-specific preprocessing function
        augment_config (Dict, optional): Augmentation configuration
        
    Returns:
        Tuple: (train_ds, test_ds, X_train, y_train, X_test, y_test)
            - train_ds: tf.data.Dataset for training
            - test_ds: tf.data.Dataset for testing
            - X_train, y_train: Raw training arrays (for evaluation)
            - X_test, y_test: Raw testing arrays (for evaluation)
    """
    logger.info("Building TensorFlow datasets...")
    
    # Prepare tensors
    X_train, y_train = prep_tensors_with_preprocess(
        train_samples, label_to_int, img_size, preprocess_fn
    )
    X_test, y_test = prep_tensors_with_preprocess(
        test_samples, label_to_int, img_size, preprocess_fn
    )
    
    # Create augmenter
    augmenter = data_augmentater(img_size, augment_config)
    
    # Build training dataset with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train), reshuffle_each_iteration=True)
    train_ds = train_ds.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build test dataset (no augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Datasets built: train={len(X_train)}, test={len(X_test)}, batch_size={batch_size}")
    
    return train_ds, test_ds, X_train, y_train, X_test, y_test


def make_split_three_way(
    dataset: List[Tuple[str, 'np.ndarray', str]], 
    val_ratio: float = 0.15,
    test_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = False
) -> Tuple[List[Tuple], List[Tuple], List[Tuple], Dict[str, int]]:
    """
    Create 3-way train/val/test split by subject (NO DATA LEAKAGE).
    
    Uses subject-independent splitting to ensure realistic benchmarking:
    - All samples from a subject stay in the same group
    - No subject appears in multiple splits
    - Prevents overly optimistic performance estimates
    
    Args:
        dataset (List[Tuple]): Full dataset of (exercise_name, image, subject_id)
        val_ratio (float): Fraction of subjects for validation (default: 0.15)
        test_ratio (float): Fraction of subjects for test (default: 0.3)
        seed (int): Random seed for reproducible splitting (default: None)
        stratified (bool): If True, ensures all exercise classes in train/val/test splits
        
    Returns:
        Tuple containing:
        - train_samples: Training samples (~55% of subjects)
        - val_samples: Validation samples (~15% of subjects)
        - test_samples: Test samples (~30% of subjects)
        - label_to_int: Label encoding dictionary
    """
    train_samples, val_samples, test_samples = split_by_subjects_three_way(
        dataset, val_ratio, test_ratio, seed=seed, stratified=stratified
    )
    
    # Create label encoding from all dataset labels (consistent across splits)
    all_labels = [item[0] for item in dataset]
    label_to_int = to_int(all_labels)
    
    logger.info(f"3-way split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    logger.info(f"Number of classes: {len(label_to_int)}")
    
    return train_samples, val_samples, test_samples, label_to_int


def make_pose_split_three_way(
    dataset: List[Tuple[str, 'np.ndarray', str, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.3,
    seed: int = None,
    stratified: bool = True
) -> Tuple[List[Tuple], List[Tuple], List[Tuple], Dict[str, int]]:
    """Subject-wise 3-way split for pose data with stratification enabled by default.

    Args:
        dataset (List[Tuple]): (exercise_name, sequence, subject_id, view) tuples.
        val_ratio (float): Fraction of subjects for validation split.
        test_ratio (float): Fraction of subjects for test split.
        seed (int): Random seed for reproducibility.
        stratified (bool): Ensure every class appears in train/val/test when possible.

    Returns:
        Tuple[List[Tuple], List[Tuple], List[Tuple], Dict[str, int]]: Train/val/test splits
        and label-to-int mapping.
    """

    train_samples, val_samples, test_samples = split_by_subjects_three_way(
        dataset, val_ratio, test_ratio, seed=seed, stratified=stratified
    )

    all_labels = [item[0] for item in dataset]
    label_to_int = to_int(all_labels)

    logger.info(
        f"Pose split (3-way): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, classes={len(label_to_int)}"
    )

    return train_samples, val_samples, test_samples, label_to_int


def build_datasets_three_way(
    train_samples: List[Tuple],
    val_samples: List[Tuple],
    test_samples: List[Tuple],
    label_to_int: Dict[str, int],
    batch_size: int = 32,
    img_size: int = 224,
    preprocess_fn = None,
    augment_config: Optional[Dict] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray']:
    """
    Build train/val/test datasets with augmentation.
    
    Creates optimized tf.data.Dataset pipelines:
    - Training: shuffled, augmented, batched, prefetched
    - Validation: batched, prefetched (NO augmentation)
    - Testing: batched, prefetched (NO augmentation)
    
    Args:
        train_samples (List[Tuple]): Training samples
        val_samples (List[Tuple]): Validation samples
        test_samples (List[Tuple]): Test samples
        label_to_int (Dict): Label to integer mapping
        batch_size (int): Batch size for all datasets
        img_size (int): Target image size
        preprocess_fn (callable, optional): Backbone-specific preprocessing
        augment_config (Dict, optional): Augmentation configuration
        
    Returns:
        Tuple: (train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Building 3-way TensorFlow datasets...")
    
    # Prepare tensors
    X_train, y_train = prep_tensors_with_preprocess(
        train_samples, label_to_int, img_size, preprocess_fn
    )
    X_val, y_val = prep_tensors_with_preprocess(
        val_samples, label_to_int, img_size, preprocess_fn
    )
    X_test, y_test = prep_tensors_with_preprocess(
        test_samples, label_to_int, img_size, preprocess_fn
    )
    
    # Create augmenter
    augmenter = data_augmentater(img_size, augment_config)
    
    # Build training dataset WITH augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train), reshuffle_each_iteration=True)
    train_ds = train_ds.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build validation dataset WITHOUT augmentation
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build test dataset WITHOUT augmentation
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Datasets built: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, batch_size={batch_size}")
    
    return train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test


def build_pose_datasets_three_way(
    train_samples: List[Tuple[str, 'np.ndarray', str, str]],
    val_samples: List[Tuple[str, 'np.ndarray', str, str]],
    test_samples: List[Tuple[str, 'np.ndarray', str, str]],
    label_to_int: Dict[str, int],
    batch_size: int = 64,
    *,
    standardize: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray', 'np.ndarray']:
    """Build train/val/test datasets for pose vectors with optional standardization.

    Args:
        train_samples (List[Tuple]): Training pose samples.
        val_samples (List[Tuple]): Validation pose samples.
        test_samples (List[Tuple]): Test pose samples.
        label_to_int (Dict[str, int]): Class-to-index mapping.
        batch_size (int): Batch size for all splits.
        standardize (bool): Standardize features using training-set mean/std.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tf.data pipelines plus the underlying numpy arrays for each split.
    """

    X_train, y_train = _pose_samples_to_arrays(train_samples, label_to_int)
    X_val, y_val = _pose_samples_to_arrays(val_samples, label_to_int)
    X_test, y_test = _pose_samples_to_arrays(test_samples, label_to_int)

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6

        def _norm(arr: 'np.ndarray') -> 'np.ndarray':
            return (arr - mean) / std

        X_train = _norm(X_train)
        X_val = _norm(X_val)
        X_test = _norm(X_test)

    def _build_ds(X, y, shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle and len(X) > 0:
            ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(X_train, y_train, shuffle=True)
    val_ds = _build_ds(X_val, y_val, shuffle=False)
    test_ds = _build_ds(X_test, y_test, shuffle=False)

    logger.info(
        "Datasets built with feature_dim=%s",
        X_train.shape[1] if len(X_train) else 0,
    )

    return train_ds, val_ds, test_ds, X_train, y_train, X_val, y_val, X_test, y_test


def _prepare_image(image_2d, img_size, color_mode: str):
    """Resize a GEI frame to the requested spatial size and channel format.

    Args:
        image_2d (np.ndarray): Single-channel GEI image.
        img_size (int): Target height/width (square resize).
        color_mode (str): Either ``'grayscale'`` or ``'rgb'`` to decide channel count.

    Returns:
        np.ndarray: Float32 tensor with shape ``(img_size, img_size, C)`` where ``C``
        is ``1`` for grayscale or ``3`` for RGB-stacked output.
    """

    if color_mode == 'grayscale':
        return _resize_to_grayscale(image_2d, img_size)
    return _resize_and_stack_to_rgb(image_2d, img_size).astype(np.float32)


def _streaming_generator_factory(
    samples: List[Tuple[str, 'np.ndarray', str]],
    label_to_int: Dict[str, int],
    img_size: int,
    color_mode: str = 'rgb'
):
    """Yield resized tensors and label ids on demand (no caching).

    Args:
        samples (List[Tuple[str, np.ndarray, str]]): (label, image, subject) tuples.
        label_to_int (Dict[str, int]): Mapping from label name to integer id.
        img_size (int): Spatial size used by :func:`_prepare_image`.
        color_mode (str): ``'rgb'`` or ``'grayscale'`` to control channel layout.

    Returns:
        Callable[[], Iterator[Tuple[np.ndarray, int]]]: Stateless generator function
        consumed by ``tf.data.Dataset.from_generator``.

    Raises:
        ValueError: If ``samples`` is empty.
    """

    if not samples:
        raise ValueError("samples list is empty")

    def generator():
        for exercise_name, image_2d, _ in samples:
            image = _prepare_image(image_2d, img_size, color_mode)
            yield image, label_to_int[exercise_name]

    return generator


def _build_streaming_dataset(
    samples: List[Tuple[str, 'np.ndarray', str]],
    label_to_int: Dict[str, int],
    batch_size: int,
    img_size: int,
    preprocess_fn = None,
    augment_config: Optional[Dict] = None,
    *,
    shuffle: bool = False,
    augment: bool = False,
    color_mode: str = 'rgb',
) -> tf.data.Dataset:
    """Create a tf.data pipeline that materializes samples only per step.

    Args:
        samples (List[Tuple[str, np.ndarray, str]]): Lazy-loaded GEI tuples.
        label_to_int (Dict[str, int]): Label-to-index mapping shared across splits.
        batch_size (int): Batch size for ``dataset.batch``.
        img_size (int): Spatial resolution used for resizing operations.
        preprocess_fn (callable, optional): Extra tensor transform applied prior to
            augmentation (e.g., backbone-specific normalization).
        augment_config (Optional[Dict]): Keyword arguments forwarded to
            :func:`data_augmentater`.
        shuffle (bool): Whether to shuffle examples per epoch.
        augment (bool): Whether to apply augmentation to each mini-batch.
        color_mode (str): ``'rgb'`` or ``'grayscale'``.

    Returns:
        tf.data.Dataset: Prefetched dataset yielding ``(image, label_id)``.
    """

    generator_fn = _streaming_generator_factory(samples, label_to_int, img_size, color_mode)

    channels = 1 if color_mode == 'grayscale' else 3
    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, channels), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(generator_fn, output_signature=output_signature)

    if shuffle:
        # Limit shuffle buffer to avoid materializing the entire dataset in memory
        buffer_size = min(len(samples), max(batch_size * 16, batch_size))
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    if preprocess_fn is not None:
        dataset = dataset.map(
            lambda image, label: (preprocess_fn(image), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if augment:
        augmenter = data_augmentater(img_size, augment_config)
        dataset = dataset.map(
            lambda image, label: (augmenter(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_datasets_three_way_streaming(
    train_samples: List[Tuple[str, 'np.ndarray', str]],
    val_samples: List[Tuple[str, 'np.ndarray', str]],
    test_samples: List[Tuple[str, 'np.ndarray', str]],
    label_to_int: Dict[str, int],
    batch_size: int = 32,
    img_size: int = 224,
    preprocess_fn = None,
    augment_config: Optional[Dict] = None,
    color_mode: str = 'rgb'
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, np.ndarray]]:
    """Build subject-aware streaming datasets (Option B) without numpy caches.

    Args:
        train_samples (List[Tuple[str, np.ndarray, str]]): Training subset tuples.
        val_samples (List[Tuple[str, np.ndarray, str]]): Validation subset tuples.
        test_samples (List[Tuple[str, np.ndarray, str]]): Test subset tuples.
        label_to_int (Dict[str, int]): Shared label mapping across splits.
        batch_size (int): Batch size used for all datasets.
        img_size (int): Target resize dimension for GEIs.
        preprocess_fn (callable, optional): Optional preprocessing transform.
        augment_config (Optional[Dict]): Config for :func:`data_augmentater`.
        color_mode (str): Color mode string forwarded to streaming builder.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, np.ndarray]]:
            Streaming train/val/test datasets plus cached label-id vectors for
            downstream metrics.
    """

    train_ds = _build_streaming_dataset(
        train_samples,
        label_to_int,
        batch_size,
        img_size,
        preprocess_fn,
        augment_config,
        shuffle=True,
        augment=True,
        color_mode=color_mode,
    )

    val_ds = _build_streaming_dataset(
        val_samples,
        label_to_int,
        batch_size,
        img_size,
        preprocess_fn,
        augment_config,
        shuffle=False,
        augment=False,
        color_mode=color_mode,
    )

    test_ds = _build_streaming_dataset(
        test_samples,
        label_to_int,
        batch_size,
        img_size,
        preprocess_fn,
        augment_config,
        shuffle=False,
        augment=False,
        color_mode=color_mode,
    )

    label_vectors = {
        'train': np.array([label_to_int[item[0]] for item in train_samples], dtype=np.int32),
        'val': np.array([label_to_int[item[0]] for item in val_samples], dtype=np.int32),
        'test': np.array([label_to_int[item[0]] for item in test_samples], dtype=np.int32),
    }

    return train_ds, val_ds, test_ds, label_vectors


def build_streaming_dataset(
    samples: List[Tuple[str, 'np.ndarray', str]],
    label_to_int: Dict[str, int],
    batch_size: int = 32,
    img_size: int = 224,
    preprocess_fn = None,
    augment_config: Optional[Dict] = None,
    *,
    shuffle: bool = False,
    augment: bool = False,
    color_mode: str = 'rgb'
) -> tf.data.Dataset:
    """Public wrapper for building a single streaming dataset.

    Args:
        samples (List[Tuple[str, np.ndarray, str]]): List of (label, image, subject) tuples.
        label_to_int (Dict[str, int]): Mapping from label string to integer id.
        batch_size (int): Number of samples per batch.
        img_size (int): Target resize dimension for both height and width.
        preprocess_fn (callable, optional): Tensor transform applied before augmentation.
        augment_config (Optional[Dict]): Parameters passed to `data_augmentater`.
        shuffle (bool): Whether to shuffle examples per epoch.
        augment (bool): Whether to apply augmentations.
        color_mode (str): "rgb" or "grayscale" to control channel count.

    Returns:
        tf.data.Dataset: Batched dataset yielding `(image, label_id)` pairs.
    """

    return _build_streaming_dataset(
        samples,
        label_to_int,
        batch_size,
        img_size,
        preprocess_fn,
        augment_config,
        shuffle=shuffle,
        augment=augment,
        color_mode=color_mode,
    )
