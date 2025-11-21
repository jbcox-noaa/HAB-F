"""
ConvLSTM2D model architecture for chlorophyll-a forecasting.

This module provides:
- ConvLSTM2D model building functions
- Model loading and saving utilities
- Architecture variants for different use cases
"""

import os
import logging
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, Conv2D, BatchNormalization, Lambda, Dropout
)
from tensorflow.keras.optimizers import Adam

from . import config


# Enable mixed precision for faster training on modern GPUs
mixed_precision.set_global_policy('mixed_float16')


def build_convlstm_model(
    input_shape: Tuple[int, int, int, int],
    filters_1: int = 32,
    filters_2: int = 32,
    kernel_size: Tuple[int, int] = (3, 3),
    learning_rate: float = 1e-4,
    loss: str = 'mse'
) -> tf.keras.Model:
    """
    Build a ConvLSTM2D model for chlorophyll-a forecasting.
    
    Architecture:
        Input (seq_len, H, W, 2) -> float32
        ConvLSTM2D(filters_1) -> tanh activation, return sequences
        BatchNormalization
        ConvLSTM2D(filters_2) -> tanh activation, final state only
        BatchNormalization
        Conv2D(1) -> tanh activation, spatial output
        Cast to float32 for loss computation
        Output (H, W, 1)
    
    Args:
        input_shape: (seq_len, H, W, channels)
            - seq_len: number of time steps
            - H, W: spatial dimensions
            - channels: typically 2 (scaled chla + mask)
        filters_1: Number of filters in first ConvLSTM layer
        filters_2: Number of filters in second ConvLSTM layer
        kernel_size: Convolutional kernel size
        learning_rate: Adam optimizer learning rate
        loss: Loss function ('mse', 'mae', etc.)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        
        # First ConvLSTM layer - processes sequences
        ConvLSTM2D(
            filters_1,
            kernel_size,
            padding='same',
            return_sequences=True,  # Return all time steps
            activation='tanh',
            name='convlstm_1'
        ),
        BatchNormalization(name='bn_1'),
        Dropout(0.2, name='dropout_1'),  # Add dropout for regularization
        
        # Second ConvLSTM layer - reduces to final prediction
        ConvLSTM2D(
            filters_2,
            kernel_size,
            padding='same',
            return_sequences=False,  # Return only final state
            activation='tanh',
            name='convlstm_2'
        ),
        BatchNormalization(name='bn_2'),
        Dropout(0.2, name='dropout_2'),  # Add dropout for regularization
        
        # Spatial convolution for final prediction
        Conv2D(
            1,  # Single channel output (chlorophyll)
            kernel_size,
            padding='same',
            activation='tanh',  # Output in [-1, 1]
            name='output_conv'
        ),
        
        # Cast back to float32 for loss computation
        Lambda(lambda x: tf.cast(x, tf.float32), name='to_float32')
    ], name='ConvLSTM_ChlaForecaster')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['mae']
    )
    
    logging.info(f"Built ConvLSTM model for input shape {input_shape}")
    logging.info(f"  Filters: {filters_1}, {filters_2}")
    logging.info(f"  Kernel size: {kernel_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Total parameters: {model.count_params():,}")
    
    return model


def build_deep_convlstm_model(
    input_shape: Tuple[int, int, int, int],
    filters: Tuple[int, int, int] = (32, 64, 32),
    kernel_size: Tuple[int, int] = (3, 3),
    learning_rate: float = 1e-4,
    loss: str = 'mse'
) -> tf.keras.Model:
    """
    Build a deeper ConvLSTM2D model with three layers.
    
    Architecture:
        Input (seq_len, H, W, 2)
        ConvLSTM2D(filters[0]) -> return sequences
        BatchNormalization
        ConvLSTM2D(filters[1]) -> return sequences
        BatchNormalization
        ConvLSTM2D(filters[2]) -> final state only
        BatchNormalization
        Conv2D(1) -> output
        Cast to float32
    
    Args:
        input_shape: (seq_len, H, W, channels)
        filters: Tuple of filter counts for the 3 ConvLSTM layers
        kernel_size: Convolutional kernel size
        learning_rate: Adam optimizer learning rate
        loss: Loss function
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        
        # First ConvLSTM layer
        ConvLSTM2D(
            filters[0],
            kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            name='convlstm_1'
        ),
        BatchNormalization(name='bn_1'),
        Dropout(0.2, name='dropout_1'),
        
        # Second ConvLSTM layer
        ConvLSTM2D(
            filters[1],
            kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            name='convlstm_2'
        ),
        BatchNormalization(name='bn_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Third ConvLSTM layer
        ConvLSTM2D(
            filters[2],
            kernel_size,
            padding='same',
            return_sequences=False,
            activation='tanh',
            name='convlstm_3'
        ),
        BatchNormalization(name='bn_3'),
        Dropout(0.2, name='dropout_3'),
        
        # Output convolution
        Conv2D(
            1,
            kernel_size,
            padding='same',
            activation='tanh',
            name='output_conv'
        ),
        
        # Cast to float32
        Lambda(lambda x: tf.cast(x, tf.float32), name='to_float32')
    ], name='DeepConvLSTM_ChlaForecaster')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['mae']
    )
    
    logging.info(f"Built deep ConvLSTM model for input shape {input_shape}")
    logging.info(f"  Filters: {filters}")
    logging.info(f"  Total parameters: {model.count_params():,}")
    
    return model


def load_model(
    model_path: str,
    compile: bool = True
) -> tf.keras.Model:
    """
    Load a saved Keras model.
    
    Args:
        model_path: Path to .keras model file
        compile: Whether to compile the model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path, compile=compile)
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"  Parameters: {model.count_params():,}")
    
    return model


def save_model(
    model: tf.keras.Model,
    model_path: str,
    overwrite: bool = True
):
    """
    Save a Keras model.
    
    Args:
        model: Keras model to save
        model_path: Path to save .keras file
        overwrite: Whether to overwrite existing file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if os.path.exists(model_path) and not overwrite:
        raise FileExistsError(f"Model already exists: {model_path}")
    
    model.save(model_path)
    logging.info(f"Saved model to {model_path}")


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string summary of the model architecture.
    
    Args:
        model: Keras model
        
    Returns:
        String summary
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


def build_model_from_config(
    input_shape: Tuple[int, int, int, int],
    model_type: str = "standard",
    **kwargs
) -> tf.keras.Model:
    """
    Build a model based on configuration.
    
    Args:
        input_shape: Input shape (seq_len, H, W, channels)
        model_type: Model architecture type ("standard" or "deep")
        **kwargs: Additional arguments passed to model builder
        
    Returns:
        Compiled Keras model
    """
    if model_type == "standard":
        return build_convlstm_model(input_shape, **kwargs)
    elif model_type == "deep":
        return build_deep_convlstm_model(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
