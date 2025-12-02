"""
Phase 7 Training Script: Supervised Training on GLERL-Labeled Patches

Trains the patch-based spectral ConvLSTM model using GLERL ground truth
measurements matched to PACE spectral data.

Training approach:
1. Load 110 training sequences (14, 11, 11, 173) with GLERL labels
2. Handle class imbalance (24.5% positive) with class weights
3. Train with binary crossentropy on center pixel predictions
4. Validate on 9 validation sequences
5. Save best model based on validation AUC

After training, the model can be transferred to the full-map architecture
for inference on the entire lake.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime

from . import config
from .model import build_patch_based_model, compile_model
from .augmentations import augment_patch_sequence, get_priority1_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data():
    """
    Load pre-processed GLERL patch sequences.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, norm_stats)
    """
    logger.info("Loading training data...")
    
    # Paths
    train_path = config.SPECTRAL_DATA_DIR / "glerl_patch_sequences_train.npz"
    val_path = config.SPECTRAL_DATA_DIR / "glerl_patch_sequences_val.npz"
    norm_path = config.SPECTRAL_DATA_DIR / "normalization_stats_train.npz"
    
    # Load training data
    train_data = np.load(train_path)
    X_train = train_data['X']  # (N, 14, 11, 11, 173)
    y_train = train_data['y']  # (N,)
    
    # Replace NaN with 0 (mask channel indicates validity)
    logger.info(f"Handling missing data (NaN values)...")
    n_nan = np.isnan(X_train).sum()
    if n_nan > 0:
        logger.info(f"  Found {n_nan:,} NaN values ({100*n_nan/X_train.size:.2f}%)")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"  Replaced with 0.0 (mask channel preserves validity information)")
    
    logger.info(f"Training data loaded: {X_train.shape}")
    logger.info(f"  Positive samples: {y_train.sum()} / {len(y_train)} ({100*y_train.mean():.1f}%)")
    
    # Load validation data
    val_data = np.load(val_path)
    X_val = val_data['X']
    y_val = val_data['y']
    
    # Replace NaN in validation data
    n_nan_val = np.isnan(X_val).sum()
    if n_nan_val > 0:
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Validation data loaded: {X_val.shape}")
    logger.info(f"  Positive samples: {y_val.sum()} / {len(y_val)} ({100*y_val.mean():.1f}%)")
    
    # Load normalization stats
    norm_stats = np.load(norm_path)
    logger.info(f"Normalization stats loaded: mean={norm_stats['mean'].shape}, std={norm_stats['std'].shape}")
    
    return X_train, y_train, X_val, y_val, norm_stats


def compute_class_weights(y_train):
    """
    Compute class weights to handle imbalanced dataset.
    
    Args:
        y_train: Training labels (N,)
        
    Returns:
        Dict with class weights {0: weight_neg, 1: weight_pos}
    """
    n_samples = len(y_train)
    n_positive = y_train.sum()
    n_negative = n_samples - n_positive
    
    # Inverse frequency weighting
    weight_negative = n_samples / (2 * n_negative)
    weight_positive = n_samples / (2 * n_positive)
    
    class_weights = {
        0: weight_negative,
        1: weight_positive
    }
    
    logger.info(f"Class weights computed:")
    logger.info(f"  Negative (0): {weight_negative:.3f}")
    logger.info(f"  Positive (1): {weight_positive:.3f}")
    logger.info(f"  Ratio: {weight_positive/weight_negative:.2f}x")
    
    return class_weights


def create_callbacks(model_name='phase7_supervised'):
    """
    Create training callbacks for checkpointing, early stopping, etc.
    
    Args:
        model_name: Name for saved models and logs
        
    Returns:
        List of keras callbacks
    """
    callbacks = []
    
    # Model checkpoint (save best model based on val_auc)
    checkpoint_path = config.MODEL_DIR / f"{model_name}_best.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=config.PHASE2_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard
    tensorboard_dir = config.TENSORBOARD_DIR / model_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)
    
    # CSV logger
    csv_path = config.LOG_DIR / f"{model_name}_training.csv"
    csv_logger = keras.callbacks.CSVLogger(
        str(csv_path),
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)
    
    logger.info(f"Callbacks created:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  TensorBoard: {tensorboard_dir}")
    logger.info(f"  CSV Log: {csv_path}")
    
    return callbacks


def compute_sample_weights(y_train, class_weights):
    """
    Compute per-sample weights for training.
    
    Args:
        y_train: Training labels (N,)
        class_weights: Dict with class weights {0: weight, 1: weight}
        
    Returns:
        sample_weights: Array of shape (N,) with per-sample weights
    """
    sample_weights = np.zeros_like(y_train, dtype=np.float32)
    sample_weights[y_train == 0] = class_weights[0]
    sample_weights[y_train == 1] = class_weights[1]
    
    logger.info(f"Sample weights computed:")
    logger.info(f"  Negative samples: weight={class_weights[0]:.3f}")
    logger.info(f"  Positive samples: weight={class_weights[1]:.3f}")
    
    return sample_weights


class AugmentedDataGenerator(keras.utils.Sequence):
    """
    Keras data generator with online augmentation.
    
    Applies augmentations on-the-fly during training, so each epoch
    sees different augmented variants of the data.
    """
    
    def __init__(
        self,
        X,
        y,
        sample_weights=None,
        batch_size=16,
        augmentation_config=None,
        shuffle=True
    ):
        """
        Args:
            X: Training sequences (N, T, H, W, C)
            y: Training labels (N,)
            sample_weights: Per-sample weights (N,)
            batch_size: Batch size
            augmentation_config: Dict with augmentation parameters
            shuffle: Whether to shuffle data each epoch
        """
        self.X = X
        self.y = y.reshape(-1, 1).astype(np.float32)  # (N, 1) for binary crossentropy
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.augmentation_config = augmentation_config or get_priority1_config()
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Generate one batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            (X_batch, y_batch) or (X_batch, y_batch, sample_weight_batch)
        """
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Prepare batch
        X_batch = np.zeros((len(batch_indices), *self.X.shape[1:]), dtype=np.float32)
        y_batch = self.y[batch_indices]
        
        # Apply augmentation to each sample
        for i, sample_idx in enumerate(batch_indices):
            X_aug, _ = augment_patch_sequence(
                self.X[sample_idx],
                self.y[sample_idx],
                self.augmentation_config
            )
            X_batch[i] = X_aug
        
        # Return with or without sample weights
        if self.sample_weights is not None:
            sample_weight_batch = self.sample_weights[batch_indices]
            return X_batch, y_batch, sample_weight_batch
        else:
            return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights=None,
    epochs=100,
    batch_size=16,
    callbacks=None,
    use_augmentation=True
):
    """
    Train the model with online augmentation.
    
    Args:
        model: Compiled keras model
        X_train: Training sequences (N, 14, 11, 11, 173)
        y_train: Training labels (N,)
        X_val: Validation sequences
        y_val: Validation labels
        class_weights: Dict with class weights (applied as sample_weight)
        epochs: Number of training epochs
        batch_size: Batch size
        callbacks: List of keras callbacks
        use_augmentation: Whether to apply online augmentation (default: True)
        
    Returns:
        Training history
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Reshape validation labels to (N, 1) for binary crossentropy
    y_val_reshaped = y_val.reshape(-1, 1).astype(np.float32)
    
    # Compute sample weights if class_weights provided
    sample_weight = None
    if class_weights is not None:
        sample_weight = compute_sample_weights(y_train, class_weights)
    
    # Create training data generator with augmentation
    if use_augmentation:
        logger.info("Using online augmentation (Priority 1 - Conservative)")
        aug_config = get_priority1_config()
        logger.info(f"  Augmentation config: {aug_config}")
        logger.info(f"  Expected augmentation factor: ~24x")
        
        train_generator = AugmentedDataGenerator(
            X_train,
            y_train,
            sample_weights=sample_weight,
            batch_size=batch_size,
            augmentation_config=aug_config,
            shuffle=True
        )
        
        # Train with generator
        history = model.fit(
            train_generator,
            validation_data=(X_val, y_val_reshaped),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        logger.info("Training without augmentation")
        
        # Reshape labels to (N, 1) for binary crossentropy
        y_train_reshaped = y_train.reshape(-1, 1).astype(np.float32)
        
        # Train without augmentation
        history = model.fit(
            X_train,
            y_train_reshaped,
            validation_data=(X_val, y_val_reshaped),
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    return history


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation set.
    
    Args:
        model: Trained keras model
        X_val: Validation sequences
        y_val: Validation labels
        
    Returns:
        Dict with evaluation metrics
    """
    logger.info("Evaluating model on validation set...")
    
    # Reshape labels
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    
    # Get predictions
    y_pred = model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Compute metrics
    results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    
    # Add confusion matrix info
    tp = ((y_val == 1) & (y_pred_binary == 1)).sum()
    tn = ((y_val == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_val == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_val == 1) & (y_pred_binary == 0)).sum()
    
    results['true_positives'] = int(tp)
    results['true_negatives'] = int(tn)
    results['false_positives'] = int(fp)
    results['false_negatives'] = int(fn)
    
    logger.info("Validation Results:")
    logger.info(f"  Loss: {results['loss']:.4f}")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall: {results['recall']:.4f}")
    logger.info(f"  AUC: {results['auc']:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    return results


def save_results(results, history, model_name='phase7_supervised'):
    """
    Save training results to JSON.
    
    Args:
        results: Dict with evaluation metrics
        history: Training history object
        model_name: Name for results file
    """
    results_path = config.OUTPUT_DIR / f"{model_name}_results.json"
    
    # Combine results
    output = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seq_len': config.SEQ_LEN,
            'patch_size': config.PATCH_SIZE,
            'n_features': config.N_INPUT_FEATURES,
            'encoder_layers': config.ENCODER_LAYERS,
            'convlstm_filters': config.CONVLSTM_FILTERS,
            'decoder_layers': config.DECODER_LAYERS,
            'epochs': config.PHASE2_EPOCHS,
            'batch_size': config.PHASE2_BATCH_SIZE,
            'learning_rate': config.PHASE2_LR,
        },
        'validation_metrics': {k: float(v) if not isinstance(v, int) else v for k, v in results.items()},
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'auc': [float(x) for x in history.history['auc']],
            'val_auc': [float(x) for x in history.history['val_auc']],
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


def main():
    """Main training function."""
    
    logger.info("=" * 80)
    logger.info("PHASE 7: SUPERVISED TRAINING ON GLERL-LABELED PATCHES")
    logger.info("=" * 80)
    
    # Load data
    X_train, y_train, X_val, y_val, norm_stats = load_training_data()
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Build model
    logger.info("\nBuilding model...")
    model = build_patch_based_model(
        seq_len=config.SEQ_LEN,
        patch_size=config.PATCH_SIZE,
        n_features=config.N_INPUT_FEATURES,
        encoder_output=config.ENCODER_LAYERS[-1],  # 16
        convlstm_filters=config.CONVLSTM_FILTERS,
        convlstm_kernel_size=config.CONVLSTM_KERNEL_SIZE,
    )
    
    # Compile model
    logger.info("Compiling model...")
    model = compile_model(
        model,
        learning_rate=config.PHASE2_LR
    )
    
    logger.info(f"\nModel summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_name='phase7_supervised')
    
    # Train
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        epochs=config.PHASE2_EPOCHS,
        batch_size=config.PHASE2_BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Evaluate
    results = evaluate_model(model, X_val, y_val)
    
    # Save results
    save_results(results, history, model_name='phase7_supervised')
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best model saved to: {config.MODEL_DIR / 'phase7_supervised_best.keras'}")
    logger.info(f"Results saved to: {config.OUTPUT_DIR / 'phase7_supervised_results.json'}")
    
    return model, history, results


if __name__ == '__main__':
    # Setup GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        logger.info(f"GPU available: {physical_devices}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        logger.info("No GPU available, using CPU")
    
    # Run training
    model, history, results = main()
