"""
Image preprocessing utilities for GEI (Gait Energy Image) datasets.

This module handles image resizing, RGB conversion, tensor preparation,
and label encoding for training deep learning models.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def _resize_and_stack_to_rgb(image_2d: np.ndarray, img_size: int = 224) -> np.ndarray:
    """
    Resize grayscale image and stack to 3 channels (RGB).
    
    Args:
        image_2d (np.ndarray): Input grayscale image (H, W)
        img_size (int): Target size for both dimensions
        
    Returns:
        np.ndarray: RGB image with shape (img_size, img_size, 3)
    """
    # Ensure img_size is an integer (OpenCV requires int type)
    img_size = int(img_size)
    
    # Ensure the image is in the correct format for OpenCV
    if image_2d.dtype != np.uint8:
        # Convert to uint8 if needed
        if image_2d.max() <= 1.0:
            # Scale [0, 1] to [0, 255]
            image_2d = (image_2d * 255).astype(np.uint8)
        else:
            # Already in [0, 255] range
            image_2d = image_2d.astype(np.uint8)
    
    resized = cv2.resize(image_2d, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return np.stack([resized, resized, resized], axis=-1)


def _resize_to_grayscale(image_2d: np.ndarray, img_size: int = 224) -> np.ndarray:
    """Resize grayscale image and add explicit channel dimension."""

    img_size = int(img_size)

    if image_2d.dtype != np.uint8:
        if image_2d.max() <= 1.0:
            image_2d = (image_2d * 255).astype(np.uint8)
        else:
            image_2d = image_2d.astype(np.uint8)

    resized = cv2.resize(image_2d, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=-1)


def prep_tensors_with_preprocess(
    samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    img_size: int = 224,
    preprocess_fn = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare image tensors with optional backbone-specific preprocessing.
    
    This function converts grayscale GEI images to RGB format, resizes them,
    and applies optional preprocessing functions (e.g., ImageNet normalization).
    
    Args:
        samples (List[Tuple]): List of (exercise_name, image, subject_id) tuples
        label_to_int (Dict): Mapping from exercise names to integer labels
        img_size (int): Target image size
        preprocess_fn (callable, optional): Preprocessing function to apply (e.g., mobilenet_preprocess)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (images, labels)
            - images: shape (N, img_size, img_size, 3), dtype float32
            - labels: shape (N,), dtype int32
            
    Raises:
        ValueError: If samples list is empty
    """
    if not samples:
        raise ValueError("samples list is empty")
    
    images_list = []
    labels_list = []
    
    for exercise_name, image_2d, subject_id in samples:
        rgb_img = _resize_and_stack_to_rgb(image_2d, img_size)
        images_list.append(rgb_img)
        labels_list.append(label_to_int[exercise_name])
    
    X = np.array(images_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    # Apply backbone-specific preprocessing if provided
    if preprocess_fn is not None:
        X = preprocess_fn(X)
    
    return X, y


def prep_tensors(
    samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    img_size: int = 224
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare tensors with simple normalization (divide by 255).
    
    Use this for models that expect [0, 1] normalized input without
    ImageNet-specific preprocessing.
    
    Args:
        samples (List[Tuple]): List of samples
        label_to_int (Dict): Label mapping
        img_size (int): Target size
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (images, labels)
    """
    X, y = prep_tensors_with_preprocess(samples, label_to_int, img_size, preprocess_fn=None)
    X = X / 255.0
    return X, y


def prep_tensors_grayscale(
    samples: List[Tuple[str, np.ndarray, str]],
    label_to_int: Dict[str, int],
    img_size: int = 224
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare grayscale tensors scaled to [0,1] with explicit channel axis."""

    if not samples:
        raise ValueError("samples list is empty")

    images_list = []
    labels_list = []

    for exercise_name, image_2d, _ in samples:
        image = _resize_to_grayscale(image_2d, img_size)
        images_list.append(image)
        labels_list.append(label_to_int[exercise_name])

    X = np.stack(images_list).astype(np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y


def to_int(label_strings: List[str]) -> Dict[str, int]:
    """
    Create label to integer mapping from a list of label strings.
    
    Labels are sorted alphabetically before mapping to ensure consistent
    integer assignment across different runs.
    
    Args:
        label_strings (List[str]): List of label strings (may contain duplicates)
        
    Returns:
        Dict[str, int]: Mapping from label string to integer index
        
    Example:
        >>> to_int(['cat', 'dog', 'cat', 'bird'])
        {'bird': 0, 'cat': 1, 'dog': 2}
    """
    unique_labels = sorted(set(label_strings))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_int
