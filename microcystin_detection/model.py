"""
CNN model architecture for microcystin detection from satellite imagery.

This module defines a dual-input CNN that combines:
1. Spatial patch features (hyperspectral reflectance)
2. Context features (temporal, geographic, environmental metadata)
"""

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from . import config


def build_model(
    patch_size: int,
    n_channels: int,
    context_size: int,
    learning_rate: float = 1e-4
) -> Model:
    """
    Build dual-input CNN for microcystin detection.
    
    Architecture:
    - Patch branch: Conv2D → BN → Conv2D → BN → Dense → MaxPooling
    - Context branch: Dense → Dropout
    - Combined: Concatenate → Dense layers → Sigmoid output
    
    Args:
        patch_size: Spatial dimension of patch (pixels per side)
        n_channels: Number of spectral channels (172 for PACE, 21 for Sentinel-3)
        context_size: Dimension of context feature vector
        learning_rate: Adam optimizer learning rate
        
    Returns:
        Compiled Keras Model
    """
    # ===== PATCH BRANCH =====
    # Input: (patch_size, patch_size, n_channels + 1)
    # +1 for binary mask channel indicating valid pixels
    input_patch = Input(shape=(patch_size, patch_size, n_channels + 1), name='patch_input')
    
    # First conv block: 32 filters, 1x1 kernel (spectral mixing)
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same', name='conv1')(input_patch)
    x = layers.BatchNormalization(name='bn1')(x)
    
    # Second conv block: 64 filters, 1x1 kernel
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    # Reshape to (patch_size² × 64) for pixel-wise dense layer
    x = layers.Reshape((patch_size * patch_size, 64), name='reshape')(x)
    
    # Pixel-wise dense layer
    x = layers.Dense(128, activation='relu', name='pixel_dense')(x)
    x = layers.Dropout(0.1, name='pixel_dropout')(x)
    
    # Global max pooling across spatial dimension
    x = layers.GlobalMaxPooling1D(name='global_pool')(x)
    
    # ===== CONTEXT BRANCH =====
    # Input: (context_size,)
    # Contains: lat, lon, doy, month, mean reflectance per channel, etc.
    input_context = Input(shape=(context_size,), name='context_input')
    
    c = layers.Dense(64, activation='relu', name='context_dense')(input_context)
    c = layers.Dropout(0.1, name='context_dropout')(c)
    
    # ===== COMBINED BRANCH =====
    # Concatenate patch features (128-dim) and context features (64-dim)
    merged = layers.Concatenate(name='merge')([x, c])
    
    # Progressive dimensionality reduction
    for i, units in enumerate([64, 32, 4], start=1):
        merged = layers.Dense(units, activation='relu', name=f'dense{i}')(merged)
        merged = layers.BatchNormalization(name=f'bn_dense{i}')(merged)
        merged = layers.Dropout(0.1, name=f'dropout_dense{i}')(merged)
    
    # Binary classification output
    output = layers.Dense(1, activation='sigmoid', name='output')(merged)
    
    # ===== COMPILE MODEL =====
    model = Model(
        inputs=[input_patch, input_context],
        outputs=output,
        name='microcystin_cnn'
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    logging.info(f"Built CNN model: patch_size={patch_size}, channels={n_channels}, "
                 f"context_size={context_size}")
    
    return model


def get_model_config(sensor: str, patch_size: int) -> dict:
    """
    Get model configuration for a given sensor and patch size.
    
    Args:
        sensor: Sensor name ('PACE' or 'Sentinel-3')
        patch_size: Patch dimension (e.g., 3, 5, 7, 9)
        
    Returns:
        Dictionary with model configuration
    """
    sensor_params = config.SENSOR_PARAMS[sensor]
    n_channels = len(sensor_params['channels'])
    
    # Context features: lat, lon, doy, month, hour, mean_reflectance_per_channel
    context_size = 5 + n_channels
    
    return {
        'patch_size': patch_size,
        'n_channels': n_channels,
        'context_size': context_size,
        'sensor': sensor,
        'learning_rate': config.LEARNING_RATE
    }


def load_model_with_normalization(model_path: str, stats_dir: str) -> Tuple[Model, dict]:
    """
    Load saved model and its normalization statistics.
    
    Args:
        model_path: Path to saved .keras model file
        stats_dir: Directory containing normalization stats (.npy files)
        
    Returns:
        Tuple of (model, normalization_stats)
        - model: Loaded Keras Model
        - normalization_stats: Dict with 'patch_means', 'patch_stds', 
                              'context_means', 'context_stds'
    """
    import os
    import numpy as np
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Loaded model from {model_path}")
    
    # Load normalization stats
    stats = {}
    try:
        stats['patch_means'] = np.load(os.path.join(stats_dir, 'channel_stats', 'means.npy'))
        stats['patch_stds'] = np.load(os.path.join(stats_dir, 'channel_stats', 'stds.npy'))
        stats['context_means'] = np.load(os.path.join(stats_dir, 'context_means.npy'))
        stats['context_stds'] = np.load(os.path.join(stats_dir, 'context_stds.npy'))
        logging.info(f"Loaded normalization stats from {stats_dir}")
    except FileNotFoundError as e:
        logging.warning(f"Could not load normalization stats: {e}")
        stats = None
    
    return model, stats
