"""
Training pipeline for microcystin probability forecasting.

This module handles:
- Loading MC probability sequences
- Training ConvLSTM2D model
- Model checkpointing and early stopping
- Training history visualization and logging
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

from . import config
from .utils import configure_logging, load_mc_sequences, save_plot
from .model import build_mc_convlstm_model, save_model_checkpoint


# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


def build_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from NumPy arrays.
    
    Args:
        X: Input sequences (N, seq_len, H, W, 1)
        y: Target maps (N, H, W, 1)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        TensorFlow Dataset optimized for training
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 1000))
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds


def plot_training_history(
    history,
    output_path: Optional[str] = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Keras training history object
        output_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    ax2.plot(epochs, history.history['mae'], 'b-', label='Training MAE', linewidth=2)
    ax2.plot(epochs, history.history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved training history plot to {output_path}")
    else:
        save_plot(fig, basename='mc_training_history')
    
    plt.close()


def train_mc_forecasting_model(
    data_dir: str = None,
    output_dir: str = None,
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None,
    patience: int = None
):
    """
    Main training pipeline for MC probability forecasting.
    
    This function:
    1. Loads MC probability sequences (with temporal split)
    2. Builds ConvLSTM model
    3. Sets up training callbacks (checkpointing, early stopping, etc.)
    4. Trains model with validation monitoring
    5. Evaluates on test set
    6. Saves model and training history
    
    Args:
        data_dir: Directory with MC probability maps (default: config.DATA_DIR)
        output_dir: Output directory for models/plots (default: config.MODEL_DIR)
        batch_size: Training batch size (default: config.BATCH_SIZE)
        epochs: Maximum training epochs (default: config.EPOCHS)
        learning_rate: Learning rate (default: config.LEARNING_RATE)
        patience: Early stopping patience (default: config.PATIENCE)
    """
    configure_logging()
    
    # Use config defaults if not specified
    data_dir = data_dir or str(config.DATA_DIR)
    output_dir = output_dir or str(config.MODEL_DIR)
    batch_size = batch_size or config.BATCH_SIZE
    epochs = epochs or config.EPOCHS
    learning_rate = learning_rate or config.LEARNING_RATE
    patience = patience or config.PATIENCE
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info("=" * 80)
    logging.info("MC PROBABILITY FORECASTING - TRAINING PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Random seed: {RANDOM_SEED}")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Max epochs: {epochs}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Early stopping patience: {patience}")
    
    # ========== STEP 1: LOAD DATA ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 1: LOADING MC PROBABILITY SEQUENCES")
    logging.info("=" * 80)
    
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences(
        data_dir=data_dir
    )
    
    # Replace NaN values with 0 (non-lake pixels)
    # This is necessary because NaN propagates through neural network
    logging.info("\nReplacing NaN values with 0 (non-lake pixels)...")
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    logging.info("✓ NaN values replaced with 0")
    
    logging.info(f"\nData shapes:")
    logging.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    logging.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # ========== STEP 2: BUILD DATASETS ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 2: BUILDING TF.DATA DATASETS")
    logging.info("=" * 80)
    
    ds_train = build_dataset(X_train, y_train, batch_size, shuffle=True)
    ds_val = build_dataset(X_val, y_val, batch_size, shuffle=False)
    ds_test = build_dataset(X_test, y_test, batch_size, shuffle=False)
    
    n_train_batches = int(np.ceil(len(X_train) / batch_size))
    n_val_batches = int(np.ceil(len(X_val) / batch_size))
    n_test_batches = int(np.ceil(len(X_test) / batch_size))
    
    logging.info(f"Train dataset: {len(X_train)} samples, {n_train_batches} batches")
    logging.info(f"Val dataset:   {len(X_val)} samples, {n_val_batches} batches")
    logging.info(f"Test dataset:  {len(X_test)} samples, {n_test_batches} batches")
    
    # ========== STEP 3: BUILD MODEL ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 3: BUILDING CONVLSTM MODEL")
    logging.info("=" * 80)
    
    model = build_mc_convlstm_model(learning_rate=learning_rate)
    
    # Print model summary
    print("\n")
    model.summary()
    print("\n")
    
    # ========== STEP 4: SETUP CALLBACKS ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 4: SETTING UP TRAINING CALLBACKS")
    logging.info("=" * 80)
    
    callbacks = []
    
    # 1. Model Checkpoint - save best model
    checkpoint_path = os.path.join(output_dir, 'best_model.keras')
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_cb)
    logging.info(f"✓ ModelCheckpoint: {checkpoint_path}")
    logging.info(f"  Monitor: val_loss (save best)")
    
    # 2. Early Stopping - stop if no improvement
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop_cb)
    logging.info(f"✓ EarlyStopping: patience={patience}")
    logging.info(f"  Monitor: val_loss")
    
    # 3. Reduce Learning Rate - reduce LR on plateau
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr_cb)
    logging.info(f"✓ ReduceLROnPlateau: factor=0.5, patience=5")
    logging.info(f"  Monitor: val_loss")
    
    # 4. CSV Logger - log metrics to file
    csv_log_path = os.path.join(output_dir, 'training_history.csv')
    csv_logger_cb = CSVLogger(csv_log_path)
    callbacks.append(csv_logger_cb)
    logging.info(f"✓ CSVLogger: {csv_log_path}")
    
    # ========== STEP 5: TRAIN MODEL ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 5: TRAINING MODEL")
    logging.info("=" * 80)
    logging.info(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("")
    logging.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total epochs trained: {len(history.history['loss'])}")
    
    # Get best epoch info
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = np.min(history.history['val_loss'])
    best_val_mae = history.history['val_mae'][best_epoch - 1]
    
    logging.info(f"\nBest model (epoch {best_epoch}):")
    logging.info(f"  Val Loss (MSE): {best_val_loss:.6f}")
    logging.info(f"  Val MAE: {best_val_mae:.6f}")
    
    # ========== STEP 6: EVALUATE ON TEST SET ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 6: EVALUATING ON TEST SET")
    logging.info("=" * 80)
    
    test_results = model.evaluate(ds_test, verbose=1)
    test_loss = test_results[0]
    test_mae = test_results[1]
    test_mse = test_results[2]
    
    logging.info(f"\nTest Set Performance:")
    logging.info(f"  Loss (MSE): {test_loss:.6f}")
    logging.info(f"  MAE: {test_mae:.6f}")
    logging.info(f"  MSE: {test_mse:.6f}")
    logging.info(f"  RMSE: {np.sqrt(test_mse):.6f}")
    
    # ========== STEP 7: SAVE TRAINING HISTORY PLOT ==========
    logging.info("\n" + "=" * 80)
    logging.info("STEP 7: SAVING TRAINING HISTORY PLOT")
    logging.info("=" * 80)
    
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # ========== STEP 8: SAVE FINAL SUMMARY ==========
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 80)
    
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MC PROBABILITY FORECASTING - TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Sequence length: {config.SEQ_LEN} days\n")
        f.write(f"  Forecast horizon: {config.FORECAST_HORIZON} day(s)\n")
        f.write(f"  Max gap: {config.MAX_GAP_DAYS} days\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Max epochs: {epochs}\n")
        f.write(f"  Patience: {patience}\n\n")
        
        f.write("DATA:\n")
        f.write(f"  Train sequences: {len(X_train)}\n")
        f.write(f"  Val sequences: {len(X_val)}\n")
        f.write(f"  Test sequences: {len(X_test)}\n")
        f.write(f"  Input shape: {X_train.shape[1:]}\n")
        f.write(f"  Output shape: {y_train.shape[1:]}\n\n")
        
        f.write("MODEL:\n")
        f.write(f"  Total parameters: {model.count_params():,}\n")
        f.write(f"  Architecture: 2 ConvLSTM layers (32 filters each)\n")
        f.write(f"  Output activation: sigmoid (probabilities)\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"  Epochs trained: {len(history.history['loss'])}\n")
        f.write(f"  Best epoch: {best_epoch}\n")
        f.write(f"  Best val loss: {best_val_loss:.6f}\n")
        f.write(f"  Best val MAE: {best_val_mae:.6f}\n\n")
        
        f.write("TEST SET PERFORMANCE:\n")
        f.write(f"  Loss (MSE): {test_loss:.6f}\n")
        f.write(f"  MAE: {test_mae:.6f}\n")
        f.write(f"  RMSE: {np.sqrt(test_mse):.6f}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write(f"  Best model: {checkpoint_path}\n")
        f.write(f"  Training log: {log_file}\n")
        f.write(f"  Training history CSV: {csv_log_path}\n")
        f.write(f"  Training history plot: {history_plot_path}\n")
    
    logging.info(f"Saved training summary to {summary_path}")
    
    logging.info("\n" + "=" * 80)
    logging.info("✓ TRAINING COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"Best model saved to: {checkpoint_path}")
    logging.info(f"Training log saved to: {log_file}")
    logging.info(f"Test MAE: {test_mae:.6f}")
    logging.info("=" * 80)
    
    return model, history


if __name__ == "__main__":
    """Run training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MC probability forecasting model')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory (default: from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Max epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: from config)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (default: from config)')
    
    args = parser.parse_args()
    
    # Run training
    model, history = train_mc_forecasting_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )
