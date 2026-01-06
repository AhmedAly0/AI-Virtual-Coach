"""
Data loading and preprocessing utilities for GEI, pose estimates exercise recognition experiments.
"""

from .data_loader import (
    load_data,
    split_by_subject_two_way,
    split_by_subjects_three_way,
    get_subjects_identities,
    load_front_side_geis,
    load_pose_data,
    build_subject_folds,
    verify_subject_split_integrity,
)
from .preprocessing import (
    prep_tensors_with_preprocess,
    prep_tensors,
    prep_tensors_grayscale,
    to_int,
    _resize_and_stack_to_rgb,
    _resize_to_grayscale,
)
from .augmentation import (
    data_augmentater,
    BASIC_AUGMENTATION,
    ENHANCED_AUGMENTATION
)
from .dataset_builder import (
    make_split,
    build_datasets,
    make_split_three_way,
    build_datasets_three_way,
    make_pose_split_three_way,
    build_pose_datasets_three_way,
    build_datasets_three_way_streaming,
    build_streaming_dataset,
)

# Default batch size (can be overridden)
DEFAULT_BATCH_SIZE = 32

__all__ = [
    'load_data',
    'split_by_subject_two_way',
    'split_by_subjects_three_way',
    'get_subjects_identities',
    'load_front_side_geis',
    'build_subject_folds',
    'load_pose_data',
    'prep_tensors_with_preprocess',
    'prep_tensors',
    'prep_tensors_grayscale',
    'to_int',
    '_resize_and_stack_to_rgb',
    '_resize_to_grayscale',
    'data_augmentater',
    'BASIC_AUGMENTATION',
    'ENHANCED_AUGMENTATION',
    'make_split',
    'build_datasets',
    'make_split_three_way',
    'build_datasets_three_way',
    'make_pose_split_three_way',
    'build_pose_datasets_three_way',
    'build_datasets_three_way_streaming',
    'build_streaming_dataset',
    'DEFAULT_BATCH_SIZE',
]
