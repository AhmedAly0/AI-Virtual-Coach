"""
Model architecture builders for GEI exercise recognition.

This module provides functions to build different model architectures:
- Simple CNN
- Transfer learning with standard heads (Experiment 1)
- Transfer learning with architecture-specific heads (Experiment 2)
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    VGG16, ResNet50,
    EfficientNetV2B0, EfficientNetV2B2, EfficientNetV2B3,
    MobileNetV3Large, MobileNetV2
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple, List, Dict

# Import preprocessing functions
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess

logger = logging.getLogger(__name__)


# Backbone registry mapping backbone names to (model_class, preprocess_function)
BACKBONE_REGISTRY = {
    'efficientnet_b0': (EfficientNetV2B0, efficientnetv2_preprocess),
    'efficientnet_b2': (EfficientNetV2B2, efficientnetv2_preprocess),
    'efficientnet_b3': (EfficientNetV2B3, efficientnetv2_preprocess),
    'resnet50': (ResNet50, resnet50_preprocess),
    'vgg16': (VGG16, vgg16_preprocess),
    'mobilenet_v2': (MobileNetV2, mobilenetv2_preprocess),
    'mobilenet_v3_large': (MobileNetV3Large, mobilenetv3_preprocess),
}


def build_model(
    img_size: int = 224,
    num_classes: int = 15,
    initial_lr: float = 0.001
) -> models.Model:
    """
    Build a simple CNN model for baseline experiments.
    
    Architecture:
    - 3 Convolutional blocks (32, 64, 128 filters)
    - Each block: Conv2D → MaxPooling2D
    - Dense(128) → Dropout(0.5) → Dense(num_classes)
    
    Args:
        img_size (int): Input image size
        num_classes (int): Number of output classes
        initial_lr (float): Initial learning rate
        
    Returns:
        models.Model: Compiled Keras model
    """
    logger.info(f"Building simple CNN: img_size={img_size}, classes={num_classes}")
    
    model = Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_model_for_backbone(
    backbone: str,
    img_size: int = 224,
    num_classes: int = 15,
    initial_lr: float = 0.001
) -> Tuple[models.Model, callable]:
    """
    Build transfer learning model with standard classification head (Experiment 1).
    
    Architecture:
    - Pretrained backbone (frozen initially)
    - GlobalAveragePooling2D
    - Dense(256, relu) → Dropout(0.5) → Dense(num_classes, softmax)
    
    Supported backbones:
    - efficientnet_b0/b2/b3
    - resnet50
    - vgg16
    - mobilenet_v2/v3_large
    
    Args:
        backbone (str): Backbone architecture name
        img_size (int): Input image size
        num_classes (int): Number of output classes
        initial_lr (float): Initial learning rate
        
    Returns:
        Tuple[models.Model, callable]: (compiled_model, preprocessing_function)
        
    Raises:
        ValueError: If backbone is not supported
    """
    logger.info(f"Building model with backbone: {backbone}")
    
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    BackboneClass, preprocess_fn = BACKBONE_REGISTRY[backbone]
    
    # Load pretrained backbone (frozen)
    base_model = BackboneClass(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    # Build model with standard head
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name=f"model_{backbone}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built: {model.count_params():,} total parameters")
    
    return model, preprocess_fn


def build_model_for_backbone_v2(
    backbone: str,
    img_size: int = 224,
    num_classes: int = 15,
    initial_lr: float = 0.001
) -> Tuple[models.Model, models.Model, callable]:
    """
    Build transfer learning model with ARCHITECTURE-SPECIFIC heads (Experiment 2).
    
    This version uses optimized classification heads tailored for each backbone:
    - EfficientNet: Dense(512, swish) + BN + Dropout(0.4) + Dense(256, swish) + Dropout(0.3)
    - ResNet: Dual-branch architecture with concatenation
    - VGG: Dense(1024) + BN + Dropout(0.7) + Dense(512) + Dropout(0.6)
    - MobileNet: Dense(256) + BN + Dropout(0.5)
    
    Args:
        backbone (str): Backbone architecture name
        img_size (int): Input image size
        num_classes (int): Number of output classes
        initial_lr (float): Initial learning rate
        
    Returns:
        Tuple[models.Model, models.Model, callable]: 
            (compiled_model, base_model, preprocessing_function)
            Note: Returns base_model separately for progressive unfreezing
            
    Raises:
        ValueError: If backbone is not supported
    """
    logger.info(f"Building model with OPTIMIZED HEAD for: {backbone}")
    
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    BackboneClass, preprocess_fn = BACKBONE_REGISTRY[backbone]
    
    # Load pretrained backbone (frozen)
    base_model = BackboneClass(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # ARCHITECTURE-SPECIFIC HEADS
    if 'efficientnet' in backbone:
        logger.info("  → EfficientNet-optimized head")
        x = layers.Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.Dropout(0.3)(x)
        
    elif 'resnet' in backbone:
        logger.info("  → ResNet-optimized head (dual-branch)")
        x1 = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.5)(x1)
        x2 = layers.Dense(256, activation='relu')(x1)
        x2 = layers.Dropout(0.4)(x2)
        x = layers.Concatenate()([x1, x2])
        
    elif 'vgg' in backbone:
        logger.info("  → VGG-optimized head (heavy regularization)")
        x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x)
        x = layers.Dropout(0.6)(x)
        
    elif 'mobilenet' in backbone:
        logger.info("  → MobileNet-optimized head (lightweight)")
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name=f"GEI_{backbone}_v2")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built: {model.count_params():,} total parameters")
    
    return model, base_model, preprocess_fn


def get_callbacks(
    checkpoint_path: str,
    monitor: str = 'val_loss',
    patience: int = 10,
    min_delta: float = 0.001
) -> List:
    """
    Create standard training callbacks.
    
    Returns:
    - EarlyStopping: Stop training when monitored metric stops improving
    - ReduceLROnPlateau: Reduce learning rate when metric plateaus
    - ModelCheckpoint: Save best model during training
    
    Args:
        checkpoint_path (str): Path to save model checkpoints
        monitor (str): Metric to monitor ('val_loss', 'loss', 'val_accuracy', etc.)
        patience (int): Epochs to wait before stopping/reducing LR
        min_delta (float): Minimum change to qualify as improvement
        
    Returns:
        List: List of Keras callbacks
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy' if 'val' in monitor else 'accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    logger.info(f"Callbacks created: monitoring '{monitor}', patience={patience}")
    
    return callbacks

"""
Model builder v3 - Refined architecture-specific heads for Experiment 3.

Key improvements over v2:
- All backbones use GlobalAveragePooling2D (no Flatten)
- Simpler 2-layer heads (not 1-layer universal or 3-layer complex)
- Match activations to backbone design (swish for EfficientNet, relu for others)
- Progressive dropout (higher → lower through layers)
- Scale head size to backbone capacity

Design principles:
- GlobalAvgPooling: Better for spatially-aggregated GEI patterns
- 2 Dense layers: Sufficient capacity without overfitting
- Activation matching: Align with backbone's native activation
- Progressive dropout: Higher in first layer, lower in second
- Size scaling: Larger heads for deeper backbones
"""

def build_model_for_backbone_v3(
    backbone: str,
    img_size: int = 224,
    num_classes: int = 15,
    initial_lr: float = 0.001
) -> Tuple[models.Model, callable]:
    """
    Build transfer learning model with REFINED architecture-specific heads (Experiment 3).
    
    Architecture-specific heads (all use GlobalAveragePooling2D):
    
    EfficientNet family:
        GlobalAvgPool → Dense(512, swish) + BN + Dropout(0.3) 
                     → Dense(256, swish) + Dropout(0.2) → Output
    
    ResNet50:
        GlobalAvgPool → Dense(1024, relu) + Dropout(0.4) 
                     → Dense(512, relu) + Dropout(0.3) → Output
    
    VGG16:
        GlobalAvgPool → Dense(512, relu) + Dropout(0.5) 
                     → Dense(256, relu) + Dropout(0.4) → Output
    
    MobileNet family:
        GlobalAvgPool → Dense(256, relu) + Dropout(0.25) 
                     → Dense(128, relu) + Dropout(0.15) → Output
    
    Args:
        backbone (str): Backbone architecture name
        img_size (int): Input image size
        num_classes (int): Number of output classes
        initial_lr (float): Initial learning rate
        
    Returns:
        Tuple[models.Model, callable]: (compiled_model, preprocessing_function)
        
    Raises:
        ValueError: If backbone is not supported
    """
    logger.info(f"Building model with REFINED ARCHITECTURE-SPECIFIC HEAD (v3): {backbone}")
    
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    BackboneClass, preprocess_fn = BACKBONE_REGISTRY[backbone]
    
    # Load pretrained backbone (frozen)
    base_model = BackboneClass(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    
    # ALL backbones use GlobalAveragePooling2D
    x = layers.GlobalAveragePooling2D()(x)
    
    # ARCHITECTURE-SPECIFIC 2-LAYER HEADS
    if 'efficientnet' in backbone:
        logger.info("  → EfficientNet head: Dense(512,swish)+BN+Drop(0.3) → Dense(256,swish)+Drop(0.2)")
        x = layers.Dense(512, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.Dropout(0.2)(x)
        
    elif 'resnet' in backbone:
        logger.info("  → ResNet50 head: Dense(1024,relu)+Drop(0.4) → Dense(512,relu)+Drop(0.3)")
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
    elif 'vgg' in backbone:
        logger.info("  → VGG16 head: Dense(512,relu)+Drop(0.5) → Dense(256,relu)+Drop(0.4)")
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
    elif 'mobilenet' in backbone:
        logger.info("  → MobileNet head: Dense(256,relu)+Drop(0.25) → Dense(128,relu)+Drop(0.15)")
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.15)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name=f"GEI_{backbone}_v3")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built: {model.count_params():,} total parameters")
    
    return model, preprocess_fn


def categorical_with_label_smoothing(num_classes, label_smoothing):
    """Create a categorical cross-entropy loss with smoothing for sparse labels.

    Args:
        num_classes (int): Total number of output classes used for one-hot encoding.
        label_smoothing (float): Amount of uniform smoothing applied to target vectors.

    Returns:
        Callable: Loss function that accepts sparse integer labels and model predictions.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )

    def loss_fn(y_true, y_pred):
        # Flatten potential singleton label dimensions before one-hot encoding
        y_true = tf.reshape(tf.cast(y_true, tf.int32), (-1,))
        y_true = tf.one_hot(y_true, depth=num_classes)
        return cce(y_true, y_pred)

    return loss_fn


def build_small_gei_cnn(
    img_size: int = 224,
    num_classes: int = 15,
    dense_units: int = 128,
    input_channels: int = 1,
    dropout_rate: float = 0.35
) -> models.Model:
    """Small dual-pooling CNN tailored for GEI grayscale inputs."""

    inputs = layers.Input(shape=(img_size, img_size, input_channels), name='gei_input')

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='small_gei_cnn')
    logger.info("Built small GEI CNN with %s parameters", f"{model.count_params():,}")
    return model



def build_model_for_backbone_v4(
    backbone: str,
    config: Dict
) -> Tuple[models.Model, models.Model, callable]:
    """
    Build transfer learning model with REGULARIZED heads and dual pooling (Experiment 4).
    
    Key features:
    - Dual pooling: GAP + GMP concatenated (captures both average and max patterns)
    - Fixed architecture heads (not configurable - designed for optimal performance)
    - Configurable hyperparameters: img_size, num_classes, initial_lr, label_smoothing
    - Label smoothing: Softens target labels (configurable)
    - AdamW optimizer: Uses weight decay instead of L2 regularization
    
    Architecture-specific heads (all use dual GAP+GMP pooling, hardcoded):
    
    EfficientNet family (B0/B2/B3):
        Dual pooling → BN → Dense(256, swish) + Dropout(0.3) 
                    → Dense(128, swish) + Dropout(0.2) → Output
        
    ResNet50:
        Dual pooling → BN → Dense(256, relu) + Dropout(0.4) → Output
    
    VGG16:
        Dual pooling → BN → Dense(256, relu) + Dropout(0.4) → Output
    
    MobileNet family (V2/V3-Large):
        Dual pooling → Dense(128, relu) + Dropout(0.25) → Output
    
    Args:
        backbone (str): Backbone architecture name
        config (Dict): Configuration dictionary containing:
            - model: img_size, num_classes, label_smoothing
            - training: initial_lr
        
    Returns:
        Tuple[models.Model, models.Model, callable]:
            - Model with regularized head
            - Base backbone model (for progressive unfreezing)
            - Preprocessing function
        
    Raises:
        ValueError: If backbone is not supported or required config keys are missing
        
    Notes:
        - Model compiled with Adam (AdamW with weight_decay set in training script)
        - Use differential learning rates: head vs backbone (set in training script)
        - Dual pooling doubles feature dimension after concatenation
        - Main hyperparameters loaded from config; architecture is fixed for optimal performance
    """
    # Extract parameters from config
    img_size = config['model']['img_size']
    num_classes = config['model']['num_classes']
    initial_lr = config['training']['initial_lr']
    label_smoothing = config['model']['label_smoothing']
    
    logger.info(f"Building model with REGULARIZED DUAL-POOLING HEAD (v4): {backbone}")
    logger.info(f"  Config: img_size={img_size}, lr={initial_lr}, label_smoothing={label_smoothing}")
    
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    BackboneClass, preprocess_fn = BACKBONE_REGISTRY[backbone]
    
    # Load pretrained backbone (frozen)
    base_model = BackboneClass(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    
    # DUAL POOLING: GlobalAveragePooling + GlobalMaxPooling concatenated
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
    x = layers.Concatenate(name='dual_pooling')([gap, gmp])
    
    logger.info(f"  → Dual pooling: GAP + GMP concatenated")
    
    # ARCHITECTURE-SPECIFIC REGULARIZED HEADS (hardcoded architectures)
    if 'efficientnet' in backbone:
        logger.info("  → EfficientNet head: BN → Dense(256,swish)+Drop(0.3) → Dense(128,swish)+Drop(0.2)")
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='swish', name='head_dense1')(x)
        x = layers.Dropout(0.3, name='head_dropout1')(x)
        x = layers.Dense(128, activation='swish', name='head_dense2')(x)
        x = layers.Dropout(0.2, name='head_dropout2')(x)
        
    elif 'resnet' in backbone:
        logger.info("  → ResNet50 head: BN → Dense(256,relu)+Drop(0.4)")
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu', name='head_dense1')(x)
        x = layers.Dropout(0.4, name='head_dropout1')(x)
        
    elif 'vgg' in backbone:
        logger.info("  → VGG16 head: BN → Dense(256,relu)+Drop(0.4)")
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu', name='head_dense1')(x)
        x = layers.Dropout(0.4, name='head_dropout1')(x)
        
    elif 'mobilenet' in backbone:
        logger.info("  → MobileNet head: Dense(128,relu)+Drop(0.25)")
        x = layers.Dense(128, activation='relu', name='head_dense1')(x)
        x = layers.Dropout(0.25, name='head_dropout1')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        name='output'
    )(x)
    
    model = models.Model(inputs, outputs, name=f"GEI_{backbone}_v4_regularized")
    
    # Compile with label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),  # Will be replaced by AdamW in training
        loss=categorical_with_label_smoothing(num_classes, label_smoothing),
        metrics=['accuracy']
    )
    
    logger.info(f"Model built: {model.count_params():,} total parameters")
    
    return model, base_model, preprocess_fn
