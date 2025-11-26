"""
ConvLSTM2D model architecture for microcystin probability forecasting.

This module provides:
- ConvLSTM2D model building functions
- Model loading and saving utilities
- Architecture adapted from successful chlorophyll forecasting (MSE=0.3965)
"""

import os
import logging
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, Conv2D, BatchNormalization, Lambda, Dropout, Layer
)
from tensorflow.keras.optimizers import Adam

from . import config


# Enable mixed precision for faster training on modern GPUs
mixed_precision.set_global_policy('mixed_float16')


class CastToFloat32(Layer):
    """Custom layer to cast tensor to float32 (for mixed precision compatibility)."""
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
    def get_config(self):
        return super().get_config()


def masked_mse_loss(y_true, y_pred, sentinel=-1.0):
    """
    MSE loss that ignores NaN values and sentinel values.
    
    Updated to handle sentinel values (e.g., -1.0) which indicate missing data.
    This allows the model to distinguish between:
    - Missing data (sentinel = -1)
    - Low bloom probability (valid value ≈ 0)
    
    Args:
        y_true: True values (may contain NaN for non-lake pixels or sentinel for missing)
        y_pred: Predicted values
        sentinel: Value used to indicate missing data (default: -1.0)
        
    Returns:
        Mean squared error computed only on valid pixels
    """
    # Create mask for valid pixels (not NaN and not sentinel)
    finite_mask = tf.math.is_finite(y_true)
    sentinel_mask = tf.not_equal(y_true, sentinel)
    mask = tf.logical_and(finite_mask, sentinel_mask)
    mask = tf.cast(mask, tf.float32)
    
    # Compute squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Apply mask (set invalid errors to 0)
    masked_error = tf.where(mask > 0, squared_error, 0.0)
    
    # Compute mean over valid pixels only
    sum_error = tf.reduce_sum(masked_error)
    count = tf.reduce_sum(mask)
    
    # Avoid division by zero
    mse = tf.where(count > 0, sum_error / count, 0.0)
    
    return mse


def masked_mae_loss(y_true, y_pred, sentinel=-1.0):
    """
    MAE loss that ignores NaN values and sentinel values.
    
    Updated to handle sentinel values (e.g., -1.0) which indicate missing data.
    
    Args:
        y_true: True values (may contain NaN for non-lake pixels or sentinel for missing)
        y_pred: Predicted values
        sentinel: Value used to indicate missing data (default: -1.0)
        
    Returns:
        Mean absolute error computed only on valid pixels
    """
    # Create mask for valid pixels (not NaN and not sentinel)
    finite_mask = tf.math.is_finite(y_true)
    sentinel_mask = tf.not_equal(y_true, sentinel)
    mask = tf.logical_and(finite_mask, sentinel_mask)
    mask = tf.cast(mask, tf.float32)
    
    # Compute absolute error
    abs_error = tf.abs(y_true - y_pred)
    
    # Apply mask (set invalid errors to 0)
    masked_error = tf.where(mask > 0, abs_error, 0.0)
    
    # Compute mean over valid pixels only
    sum_error = tf.reduce_sum(masked_error)
    count = tf.reduce_sum(mask)
    
    # Avoid division by zero
    mae = tf.where(count > 0, sum_error / count, 0.0)
    
    return mae


def build_mc_convlstm_model(
    input_shape: Tuple[int, int, int, int] = None,
    filters_1: int = None,
    filters_2: int = None,
    kernel_size: Tuple[int, int] = None,
    learning_rate: float = None,
    loss: str = 'mse'
) -> tf.keras.Model:
    """
    Build a ConvLSTM2D model for microcystin probability forecasting.
    
    Architecture (adapted from successful chlorophyll forecasting):
        Input (seq_len, H, W, 1) -> float32 [MC probability maps]
        ConvLSTM2D(32 filters) -> tanh activation, return sequences
        BatchNormalization
        Dropout(0.2)
        ConvLSTM2D(32 filters) -> tanh activation, final state only
        BatchNormalization
        Dropout(0.2)
        Conv2D(1 filter) -> sigmoid activation [output probability]
        Cast to float32 for loss computation
        Output (H, W, 1) -> probability map [0, 1]
    
    Key differences from chlorophyll model:
    - Input: 1 channel (MC probability) vs 2 channels (chla + mask)
    - Output activation: sigmoid (probabilities [0,1]) vs tanh (normalized [-1,1])
    - Same proven architecture that achieved MSE=0.3965 for chlorophyll
    
    Args:
        input_shape: (seq_len, H, W, channels) - default from config
            - seq_len: number of days in lookback window (default: 5)
            - H, W: spatial dimensions (84, 73 for Lake Erie)
            - channels: 1 for MC probability
        filters_1: First ConvLSTM layer filters (default: 32)
        filters_2: Second ConvLSTM layer filters (default: 32)
        kernel_size: Convolutional kernel size (default: (3, 3))
        learning_rate: Adam optimizer learning rate (default: 1e-5)
        loss: Loss function (default: 'mse' for probability regression)
        
    Returns:
        Compiled Keras model ready for training
        
    Notes:
        - Handles NaN values in lake mask (67% of pixels are non-lake)
        - Uses sigmoid activation for probability output [0, 1]
        - Mixed precision training enabled for performance
        - BatchNormalization for stable training
        - Dropout for regularization (prevent overfitting)
    """
    # Use config defaults if not specified
    input_shape = input_shape or config.INPUT_SHAPE
    filters_1 = filters_1 or config.FILTERS_1
    filters_2 = filters_2 or config.FILTERS_2
    kernel_size = kernel_size or config.KERNEL_SIZE
    learning_rate = learning_rate or config.LEARNING_RATE
    
    logging.info("Building MC probability ConvLSTM model")
    logging.info(f"  Input shape: {input_shape}")
    logging.info(f"  Filters: [{filters_1}, {filters_2}]")
    logging.info(f"  Kernel size: {kernel_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Loss: {loss}")
    
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        
        # First ConvLSTM layer - processes temporal sequences
        ConvLSTM2D(
            filters_1,
            kernel_size,
            padding='same',
            return_sequences=True,  # Return all time steps
            activation='tanh',
            name='convlstm_1'
        ),
        BatchNormalization(name='bn_1'),
        Dropout(0.2, name='dropout_1'),
        
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
        Dropout(0.2, name='dropout_2'),
        
        # Output convolution - probability map
        # Note: Using sigmoid (not tanh) because output is probability [0, 1]
        Conv2D(
            1,  # Single channel output (MC probability)
            kernel_size,
            padding='same',
            activation='sigmoid',  # Output in [0, 1] for probabilities
            name='output_conv'
        ),
        
        # Cast back to float32 for loss computation
        CastToFloat32(name='cast_output')
    ])
    
    # Compile model with masked loss functions
    optimizer = Adam(learning_rate=learning_rate)
    
    # Use custom masked loss functions to handle NaN values in lake mask
    model.compile(
        optimizer=optimizer,
        loss=masked_mse_loss,  # Custom loss that ignores NaN pixels
        metrics=[masked_mae_loss, masked_mse_loss]  # Custom metrics
    )
    
    logging.info(f"Model compiled successfully")
    logging.info(f"  Total parameters: {model.count_params():,}")
    
    return model


def build_mc_convlstm_dual_channel(
    input_shape=(5, 84, 73, 2),  # 2 channels: [probability, validity_mask]
    filters_1=32,
    filters_2=32,
    kernel_size=(3, 3),
    learning_rate=1e-5,
    loss='masked_mse',
    sentinel=-1.0
):
    """
    Build dual-channel ConvLSTM model for MC probability forecasting.
    
    This architecture explicitly separates MC probability values from their validity status,
    allowing the model to learn different patterns for original vs. gap-filled data.
    
    Args:
        input_shape: (timesteps, height, width, channels)
                    Default (5, 84, 73, 2) where channels are:
                    - Channel 0: MC probability (gap-filled with hybrid method)
                    - Channel 1: Validity mask (1=original data, 0=filled data)
        filters_1: Number of filters in first ConvLSTM layer (default: 32)
        filters_2: Number of filters in second ConvLSTM layer (default: 32)
        kernel_size: Size of convolution kernels (default: (3, 3))
        learning_rate: Adam optimizer learning rate (default: 1e-5)
        loss: Loss function ('masked_mse' or 'masked_mae')
        sentinel: Sentinel value for masking in loss functions (default: -1.0)
    
    Returns:
        Compiled Keras model
    
    Architecture:
        Input(5, 84, 73, 2) - Dual-channel sequences
        ↓
        ConvLSTM2D(32) + BN + Dropout - Process temporal patterns
        ↓
        ConvLSTM2D(32) + BN + Dropout - Refine predictions
        ↓
        Conv2D(1, sigmoid) - Output MC probability map
        ↓
        Output(84, 73, 1) - Single-channel probability
    
    Key Differences from Single-Channel Model:
        - Input has 2 channels instead of 1
        - Model can learn to weight original vs filled data differently
        - Validity mask provides explicit uncertainty information
        - More robust to data gaps and interpolation artifacts
    
    Expected Improvements:
        - Better handling of missing data regions
        - More accurate predictions near data gaps
        - Reduced error from gap-filling artifacts
        - Target: 15-20% improvement over Phase 1 sentinel approach
    """
    logging.info("Building DUAL-CHANNEL MC probability ConvLSTM model")
    logging.info(f"  Input shape: {input_shape} (2 channels: probability + mask)")
    logging.info(f"  Filters: [{filters_1}, {filters_2}]")
    logging.info(f"  Kernel size: {kernel_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Loss: {loss}")
    logging.info(f"  Sentinel value: {sentinel}")
    
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='dual_channel_input'),
        
        # First ConvLSTM layer - processes temporal sequences with validity info
        ConvLSTM2D(
            filters_1,
            kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            name='dual_convlstm_1'
        ),
        BatchNormalization(name='dual_bn_1'),
        Dropout(0.2, name='dual_dropout_1'),
        
        # Second ConvLSTM layer - refines prediction
        ConvLSTM2D(
            filters_2,
            kernel_size,
            padding='same',
            return_sequences=False,
            activation='tanh',
            name='dual_convlstm_2'
        ),
        BatchNormalization(name='dual_bn_2'),
        Dropout(0.2, name='dual_dropout_2'),
        
        # Output convolution - single channel MC probability
        Conv2D(
            1,
            kernel_size,
            padding='same',
            activation='sigmoid',  # Output in [0, 1]
            name='dual_output_conv'
        ),
        
        # Cast to float32 for loss computation
        CastToFloat32(name='dual_cast_output')
    ])
    
    # Compile with masked loss functions
    optimizer = Adam(learning_rate=learning_rate)
    
    if loss == 'masked_mse':
        loss_fn = lambda y_true, y_pred: masked_mse_loss(y_true, y_pred, sentinel=sentinel)
        loss_name = 'masked_mse'
    elif loss == 'masked_mae':
        loss_fn = lambda y_true, y_pred: masked_mae_loss(y_true, y_pred, sentinel=sentinel)
        loss_name = 'masked_mae'
    else:
        raise ValueError(f"Unknown loss function: {loss}. Use 'masked_mse' or 'masked_mae'")
    
    # Use masked MAE as additional metric
    mae_fn = lambda y_true, y_pred: masked_mae_loss(y_true, y_pred, sentinel=sentinel)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[mae_fn]
    )
    
    logging.info(f"Compiled dual-channel model with {loss_name} loss and masked MAE metric")
    logging.info(f"Total parameters: {model.count_params():,}")
    
    return model


def build_mc_convlstm_dual_channel_v2(
    input_shape=(5, 84, 73, 2),  # 2 channels: [probability, validity_mask]
    filters_1=32,
    filters_2=32,
    kernel_size=(3, 3),
    learning_rate=1e-3,
    loss='masked_mse',
    sentinel=-1.0,
    dropout_rate=0.4,
    l2_reg=0.001
):
    """
    Build IMPROVED dual-channel ConvLSTM model with stronger regularization.
    
    Key improvements over original dual-channel model:
        - L2 weight regularization to prevent overfitting
        - Increased dropout (0.4 vs 0.2) for better generalization
        - Support for gradient clipping via optimizer
        - Better default learning rate (1e-3 vs 5e-4)
    
    Args:
        input_shape: (timesteps, height, width, channels)
        filters_1: Number of filters in first ConvLSTM layer
        filters_2: Number of filters in second ConvLSTM layer
        kernel_size: Size of convolution kernels
        learning_rate: Adam optimizer learning rate
        loss: Loss function ('masked_mse' or 'masked_mae')
        sentinel: Sentinel value for masking in loss functions
        dropout_rate: Dropout rate (default: 0.4, increased from 0.2)
        l2_reg: L2 regularization coefficient (default: 0.001)
    
    Returns:
        Compiled Keras model with stronger regularization
    """
    from tensorflow.keras.regularizers import l2
    
    logging.info("Building IMPROVED DUAL-CHANNEL MC probability ConvLSTM model")
    logging.info(f"  Input shape: {input_shape} (2 channels: probability + mask)")
    logging.info(f"  Filters: [{filters_1}, {filters_2}]")
    logging.info(f"  Kernel size: {kernel_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Loss: {loss}")
    logging.info(f"  Sentinel value: {sentinel}")
    logging.info(f"  Dropout rate: {dropout_rate}")
    logging.info(f"  L2 regularization: {l2_reg}")
    
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='dual_channel_input_v2'),
        
        # First ConvLSTM layer with L2 regularization
        ConvLSTM2D(
            filters_1,
            kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name='dual_convlstm_1_v2'
        ),
        BatchNormalization(momentum=0.9, name='dual_bn_1_v2'),
        Dropout(dropout_rate, name='dual_dropout_1_v2'),
        
        # Second ConvLSTM layer with L2 regularization
        ConvLSTM2D(
            filters_2,
            kernel_size,
            padding='same',
            return_sequences=False,
            activation='tanh',
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name='dual_convlstm_2_v2'
        ),
        BatchNormalization(momentum=0.9, name='dual_bn_2_v2'),
        Dropout(dropout_rate, name='dual_dropout_2_v2'),
        
        # Output convolution with L2 regularization
        Conv2D(
            1,
            kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_regularizer=l2(l2_reg),
            name='dual_output_conv_v2'
        ),
        
        # Cast to float32 for loss computation
        CastToFloat32(name='dual_cast_output_v2')
    ])
    
    # Compile with masked loss functions and gradient clipping
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Gradient clipping to prevent exploding gradients
    )
    
    if loss == 'masked_mse':
        loss_fn = lambda y_true, y_pred: masked_mse_loss(y_true, y_pred, sentinel=sentinel)
        loss_name = 'masked_mse'
    elif loss == 'masked_mae':
        loss_fn = lambda y_true, y_pred: masked_mae_loss(y_true, y_pred, sentinel=sentinel)
        loss_name = 'masked_mae'
    else:
        raise ValueError(f"Unknown loss function: {loss}. Use 'masked_mse' or 'masked_mae'")
    
    # Use masked MAE as additional metric
    mae_fn = lambda y_true, y_pred: masked_mae_loss(y_true, y_pred, sentinel=sentinel)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[mae_fn]
    )
    
    logging.info(f"Compiled improved dual-channel model with {loss_name} loss")
    logging.info(f"Total parameters: {model.count_params():,}")
    logging.info(f"Gradient clipping: clipnorm=1.0")
    
    return model


def save_model_checkpoint(
    model: tf.keras.Model,
    filepath: str = None
) -> str:
    """
    Save model checkpoint to disk.
    
    Args:
        model: Trained Keras model
        filepath: Path to save model (default: config.MODEL_PATH)
        
    Returns:
        Path to saved model
    """
    if filepath is None:
        filepath = str(config.MODEL_PATH)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    model.save(filepath)
    logging.info(f"Model saved to {filepath}")
    
    return filepath


def load_trained_model(filepath: str = None) -> tf.keras.Model:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to saved model (default: config.MODEL_PATH)
        
    Returns:
        Loaded Keras model
    """
    if filepath is None:
        filepath = str(config.MODEL_PATH)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    # Load with custom objects for masked loss functions
    custom_objects = {
        'masked_mse_loss': masked_mse_loss,
        'masked_mae_loss': masked_mae_loss,
        'CastToFloat32': CastToFloat32
    }
    model = load_model(filepath, safe_mode=False, custom_objects=custom_objects)
    logging.info(f"Model loaded from {filepath}")
    
    return model


def print_model_summary(model: tf.keras.Model):
    """
    Print detailed model architecture summary.
    
    Args:
        model: Keras model to summarize
    """
    print("\n" + "="*80)
    print("MC PROBABILITY FORECASTING MODEL ARCHITECTURE")
    print("="*80)
    
    model.summary()
    
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"Loss function: {model.loss}")
    print(f"Optimizer: {model.optimizer.__class__.__name__}")
    print(f"Learning rate: {model.optimizer.learning_rate.numpy():.2e}")
    print("="*80)


if __name__ == "__main__":
    """Test model building."""
    logging.basicConfig(level=logging.INFO)
    
    # Build model
    model = build_mc_convlstm_model()
    
    # Print summary
    print_model_summary(model)
    
    # Test with dummy data
    import numpy as np
    
    print("\nTesting model with dummy data...")
    X_dummy = np.random.rand(2, 5, 84, 73, 1).astype('float32')
    y_pred = model.predict(X_dummy, verbose=0)
    
    print(f"✓ Input shape: {X_dummy.shape}")
    print(f"✓ Output shape: {y_pred.shape}")
    print(f"✓ Output range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"✓ Model works correctly!")
