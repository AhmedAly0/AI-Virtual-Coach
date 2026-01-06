"""
Data augmentation utilities for GEI (Gait Energy Image) training.

This module provides augmentation pipelines for different experiment configurations.
Experiment 1 uses basic augmentation, while Experiment 2 uses enhanced augmentation
with additional transformations.
"""

import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Optional, Dict


class RandomErasing(layers.Layer):
    """Randomly erases a rectangular region from the input image."""

    def __init__(
        self,
        probability: float = 0.2,
        min_area: float = 0.02,
        max_area: float = 0.2,
        aspect_ratio: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio = aspect_ratio

    def call(self, inputs, training=None):
        inputs = tf.convert_to_tensor(inputs)
        if inputs.shape.rank is None:
            inputs.set_shape(tf.TensorShape([None, None, None]))

        if training is None:
            training = tf.keras.backend.learning_phase()

        def erased():
            static_rank = inputs.shape.rank

            if static_rank == 4:
                return tf.map_fn(self._erase_image, inputs)
            if static_rank == 3:
                return self._erase_image(inputs)

            dynamic_rank = tf.rank(inputs)
            return tf.cond(
                tf.equal(dynamic_rank, 4),
                lambda: tf.map_fn(self._erase_image, inputs),
                lambda: self._erase_image(inputs),
            )

        return tf.cond(tf.cast(training, tf.bool), erased, lambda: inputs)

    def _erase_image(self, image):
        image = tf.convert_to_tensor(image)
        squeeze_channel = False
        if image.shape.rank == 2:
            image = tf.expand_dims(image, -1)
            squeeze_channel = True

        rnd = tf.random.uniform([])

        def perform_erase():
            shape = tf.shape(image)
            height = tf.cast(shape[0], tf.float32)
            width = tf.cast(shape[1], tf.float32)
            channels = tf.cast(shape[2], tf.int32)

            area = height * width
            target_area = tf.random.uniform([], self.min_area, self.max_area) * area
            aspect_ratio = tf.random.uniform([], self.aspect_ratio, 1.0 / self.aspect_ratio)

            erase_h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
            erase_w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

            erase_h = tf.clip_by_value(erase_h, 1, tf.cast(height, tf.int32))
            erase_w = tf.clip_by_value(erase_w, 1, tf.cast(width, tf.int32))

            max_y = tf.cast(height, tf.int32) - erase_h
            max_x = tf.cast(width, tf.int32) - erase_w
            y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
            x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)

            zeros_patch = tf.zeros((erase_h, erase_w, channels), dtype=image.dtype)
            top = image[:y]
            middle = image[y:y + erase_h]
            bottom = image[y + erase_h:]

            left = middle[:, :x]
            right = middle[:, x + erase_w:]
            middle_replaced = tf.concat([left, zeros_patch, right], axis=1)
            erased = tf.concat([top, middle_replaced, bottom], axis=0)
            return erased

        erased_image = tf.cond(rnd < self.probability, perform_erase, lambda: image)
        if squeeze_channel:
            erased_image = tf.squeeze(erased_image, axis=-1)
        return erased_image

    def get_config(self):
        config = super().get_config()
        config.update({
            'probability': self.probability,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'aspect_ratio': self.aspect_ratio,
        })
        return config

logger = logging.getLogger(__name__)


# Augmentation configurations for different experiments
BASIC_AUGMENTATION = {
    'horizontal_flip': True,
    'translation_height': 0.15,
    'translation_width': 0.15,
    'rotation': False,
    'rotation_factor': 0.1,
    'zoom': False,
    'zoom_factor': 0.1,
    'brightness': False,
    'brightness_factor': 0.1,
}

ENHANCED_AUGMENTATION = {
    'horizontal_flip': True,
    'translation_height': 0.15,
    'translation_width': 0.15,
    'rotation': True,           # Enhanced: rotation enabled
    'rotation_factor': 0.05,    # ±18 degrees
    'zoom': True,               # Enhanced: zoom enabled
    'zoom_factor': 0.1,         # ±10%
    'brightness': True,         # Enhanced: brightness enabled
    'brightness_factor': 0.1,
}


def data_augmentater(
    img_size: int = 224,
    augment_config: Optional[Dict] = None
) -> Sequential:
    """
    Create a data augmentation pipeline based on configuration.
    
    Supports various augmentation techniques:
    - Horizontal flip
    - Random translation (height and width)
    - Random rotation
    - Random zoom
    - Random brightness adjustment
    
    Args:
        img_size (int): Target image size for resizing
        augment_config (Dict, optional): Augmentation configuration dict.
            If None, uses BASIC_AUGMENTATION by default.
            
    Returns:
        Sequential: Keras Sequential model with augmentation layers
        
    Example:
        >>> # Basic augmentation
        >>> augmenter = data_augmentater(224, BASIC_AUGMENTATION)
        >>> 
        >>> # Enhanced augmentation
        >>> augmenter = data_augmentater(224, ENHANCED_AUGMENTATION)
    """
    if augment_config is None:
        augment_config = BASIC_AUGMENTATION.copy()
        logger.info("Using BASIC_AUGMENTATION (default)")
    else:
        augment_config = augment_config.copy()  # Don't modify original
    
    logger.info(f"Creating augmenter: {augment_config}")
    
    aug_layers = [layers.Resizing(img_size, img_size)]
    
    # Horizontal flip
    if augment_config.get('horizontal_flip', False):
        aug_layers.append(layers.RandomFlip("horizontal"))
        logger.debug("  + Horizontal flip enabled")
    
    # Translation
    if augment_config.get('translation_height', 0) > 0 or augment_config.get('translation_width', 0) > 0:
        aug_layers.append(
            layers.RandomTranslation(
                height_factor=augment_config.get('translation_height', 0.15),
                width_factor=augment_config.get('translation_width', 0.15),
                fill_mode='constant',
                fill_value=0.0
            )
        )
        logger.debug(f"  + Translation: height={augment_config.get('translation_height')}, "
                    f"width={augment_config.get('translation_width')}")
    
    # Rotation
    rotation_flag = augment_config.get('rotation', False) or 'rotation_degrees' in augment_config
    if rotation_flag:
        rotation_factor = augment_config.get('rotation_factor')
        if rotation_factor is None:
            degrees = augment_config.get('rotation_degrees', 0)
            rotation_factor = degrees / 360.0
        aug_layers.append(
            layers.RandomRotation(
                factor=rotation_factor,
                fill_mode='constant',
                fill_value=0.0
            )
        )
        degrees = rotation_factor * 360
        logger.debug(f"  + Rotation: ±{degrees:.1f} degrees")
    
    # Zoom
    zoom_flag = augment_config.get('zoom', False) or ('zoom_min' in augment_config and 'zoom_max' in augment_config)
    if zoom_flag:
        if 'zoom_min' in augment_config and 'zoom_max' in augment_config:
            zoom_min = augment_config['zoom_min']
            zoom_max = augment_config['zoom_max']
            zoom_layer = layers.RandomZoom(
                height_factor=(zoom_min - 1.0, zoom_max - 1.0),
                width_factor=(zoom_min - 1.0, zoom_max - 1.0),
                fill_mode='constant',
                fill_value=0.0
            )
            logger.debug(f"  + Zoom range: [{zoom_min}, {zoom_max}]")
        else:
            zoom_factor = augment_config.get('zoom_factor', 0.1)
            zoom_layer = layers.RandomZoom(
                height_factor=(-zoom_factor, 0),
                fill_mode='constant',
                fill_value=0.0
            )
            logger.debug(f"  + Zoom: ±{zoom_factor*100}%")
        aug_layers.append(zoom_layer)
    
    # Brightness
    if augment_config.get('brightness', False):
        brightness_factor = augment_config.get('brightness_factor', 0.1)
        aug_layers.append(
            layers.RandomBrightness(brightness_factor)
        )
        logger.debug(f"  + Brightness: ±{brightness_factor}")

    # Random erasing (custom layer)
    random_erasing_cfg = augment_config.get('random_erasing', {})
    if random_erasing_cfg.get('enabled', False):
        aug_layers.append(
            RandomErasing(
                probability=random_erasing_cfg.get('probability', 0.2),
                min_area=random_erasing_cfg.get('min_area', 0.02),
                max_area=random_erasing_cfg.get('max_area', 0.2),
                aspect_ratio=random_erasing_cfg.get('aspect_ratio', 0.3),
            )
        )
        logger.debug("  + Random erasing enabled")
    
    # Determine augmentation name based on config
    is_enhanced = (rotation_flag or zoom_flag or 
                   augment_config.get('brightness', False) or 
                   random_erasing_cfg.get('enabled', False))
    aug_name = "data_augmentation_enhanced" if is_enhanced else "data_augmentation_basic"
    
    return Sequential(aug_layers, name=aug_name)
