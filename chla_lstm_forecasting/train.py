"""
Training pipeline for chlorophyll-a forecasting model.

This module handles:
- Loading and preprocessing time series data
- Creating temporal sequences
- Training ConvLSTM2D models
- Model checkpointing and early stopping
- Training history visualization
"""

import os
import sys
import argparse
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from . import config
from .utils import (
    configure_logging,
    load_composite_data,
    create_sequences,
    split_temporal_data,
    validate_data,
    save_plot
)
from .model import build_model_from_config


# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)


def build_dataset_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 4,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from NumPy arrays.
    
    Args:
        X: Input sequences (num_samples, seq_len, H, W, channels)
        y: Target frames (num_samples, H, W, channels)
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        TensorFlow Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds


def plot_training_history(
    history,
    output_path: Optional[str] = None
):
    """
    Plot training history (loss curves).
    
    Args:
        history: Keras training history object
        output_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation loss
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=config.PLOT_DPI)
        logging.info(f"Saved training history to {output_path}")
    else:
        save_plot(fig, basename='training_history')
    
    plt.close()


def train_model(
    data_dir: str,
    sensor: str = "S3",
    seq_len: int = None,
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None,
    model_type: str = "standard",
    limit_files: Optional[int] = None,
    output_dir: Optional[str] = None
):
    """
    Main training pipeline.
    
    Args:
        data_dir: Directory containing composite data files
        sensor: Sensor identifier ("S3" or "PACE")
        seq_len: Sequence length (default from config)
        batch_size: Batch size (default from config)
        epochs: Number of training epochs (default from config)
        learning_rate: Learning rate (default from config)
        model_type: Model architecture ("standard" or "deep")
        limit_files: Limit number of files to use (for testing)
        output_dir: Output directory for models and plots
    """
    configure_logging()
    
    # Use config defaults if not specified
    if seq_len is None:
        seq_len = config.SEQUENCE_LENGTH
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if epochs is None:
        epochs = config.EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if output_dir is None:
        output_dir = str(config.BASE_DIR)
    
    logging.info("=" * 70)
    logging.info("CHLOROPHYLL-A FORECASTING - TRAINING PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Random seed: {config.RANDOM_SEED}")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Sensor: {sensor}")
    logging.info(f"Sequence length: {seq_len}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Model type: {model_type}")
    
    # ===== LOAD DATA =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 1: LOADING DATA")
    logging.info("=" * 70)
    
    data_files, meta_files = load_composite_data(
        data_dir,
        sensor=sensor,
        limit=limit_files
    )
    
    if len(data_files) < config.MIN_TRAINING_IMAGES:
        raise ValueError(
            f"Not enough images for training: {len(data_files)} "
            f"(minimum: {config.MIN_TRAINING_IMAGES})"
        )
    
    # ===== CREATE SEQUENCES =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 2: CREATING SEQUENCES")
    logging.info("=" * 70)
    
    X, y = create_sequences(data_files, seq_len=seq_len)
    
    # Validate data
    validate_data(X, y)
    
    # ===== SPLIT DATA =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 3: SPLITTING DATA")
    logging.info("=" * 70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_temporal_data(
        X, y,
        train_frac=config.TRAIN_SPLIT,
        val_frac=config.VAL_SPLIT
    )
    
    # ===== BUILD DATASETS =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 4: BUILDING TF.DATA DATASETS")
    logging.info("=" * 70)
    
    ds_train = build_dataset_from_numpy(X_train, y_train, batch_size, shuffle=True)
    ds_val = build_dataset_from_numpy(X_val, y_val, batch_size, shuffle=False)
    ds_test = build_dataset_from_numpy(X_test, y_test, batch_size, shuffle=False)
    
    logging.info(f"Train dataset: {len(X_train)} samples, {len(X_train)//batch_size} batches")
    logging.info(f"Val dataset: {len(X_val)} samples, {len(X_val)//batch_size} batches")
    logging.info(f"Test dataset: {len(X_test)} samples, {len(X_test)//batch_size} batches")
    
    # ===== BUILD MODEL =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 5: BUILDING MODEL")
    logging.info("=" * 70)
    
    # Infer input shape from training data
    _, seq, H, W, C = X_train.shape
    input_shape = (seq, H, W, C)
    
    logging.info(f"Input shape: {input_shape}")
    logging.info(f"  Sequence length: {seq}")
    logging.info(f"  Spatial dimensions: {H} × {W}")
    logging.info(f"  Channels: {C}")
    
    model = build_model_from_config(
        input_shape=input_shape,
        model_type=model_type,
        learning_rate=learning_rate,
        loss=config.LOSS_FUNCTION
    )
    
    # Print model summary
    model.summary()
    
    # ===== SETUP CALLBACKS =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 6: SETTING UP CALLBACKS")
    logging.info("=" * 70)
    
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, 'best_model.keras')
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
        verbose=config.CHECKPOINT_VERBOSE
    )
    callbacks.append(checkpoint_cb)
    logging.info(f"Checkpoint: {checkpoint_path}")
    
    # Early stopping
    early_stop_cb = EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
        verbose=config.EARLY_STOPPING_VERBOSE
    )
    callbacks.append(early_stop_cb)
    logging.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    # ===== TRAIN MODEL =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 7: TRAINING MODEL")
    logging.info("=" * 70)
    
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # ===== EVALUATE MODEL =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 8: EVALUATING MODEL")
    logging.info("=" * 70)
    
    test_loss, test_mae = model.evaluate(ds_test, verbose=1)
    logging.info(f"\nTest Results:")
    logging.info(f"  Loss (MSE): {test_loss:.6f}")
    logging.info(f"  MAE: {test_mae:.6f}")
    
    # ===== SAVE FINAL MODEL =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 9: SAVING FINAL MODEL")
    logging.info("=" * 70)
    
    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)
    logging.info(f"Saved final model to {final_model_path}")
    
    # ===== PLOT TRAINING HISTORY =====
    logging.info("\n" + "=" * 70)
    logging.info("STEP 10: PLOTTING TRAINING HISTORY")
    logging.info("=" * 70)
    
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # ===== SUMMARY =====
    logging.info("\n" + "=" * 70)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    logging.info(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    logging.info(f"Test loss: {test_loss:.6f}")
    logging.info(f"Models saved to: {output_dir}")
    logging.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train chlorophyll-a forecasting model'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing composite data files'
    )
    
    parser.add_argument(
        '--sensor',
        type=str,
        default='S3',
        choices=['S3', 'PACE'],
        help='Sensor identifier (default: S3)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=None,
        help=f'Sequence length (default: {config.SEQUENCE_LENGTH})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {config.EPOCHS})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='standard',
        choices=['standard', 'deep'],
        help='Model architecture (default: standard)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of data files (for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: {config.BASE_DIR})'
    )
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        sensor=args.sensor,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        limit_files=args.limit,
        output_dir=args.output_dir
    )
