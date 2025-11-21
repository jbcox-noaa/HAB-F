"""
Training pipeline for microcystin detection CNN.

This module handles:
- Loading balanced training data
- Data augmentation (spatial flips)
- Normalization and preprocessing
- Model training with callbacks
- Evaluation on test set
"""

import os
import logging
from typing import Tuple
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from . import config
from .model import build_model


def load_training_data(
    data_dir: str,
    sensor: str = 'PACE'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load balanced training data from .npy file.
    
    Args:
        data_dir: Directory containing training_data_balanced_{sensor}.npy
        sensor: Sensor name ('PACE' or 'Sentinel-3')
        
    Returns:
        Tuple of (raw_data, metadata)
        - raw_data: Array of training samples
        - metadata: Dictionary with data statistics
    """
    filepath = os.path.join(data_dir, f'training_data_balanced_{sensor}.npy')
    raw = np.load(filepath, allow_pickle=True)
    logging.info(f"Loaded {len(raw)} samples from {filepath}")
    
    # Extract metadata
    metadata = {
        'n_samples': len(raw),
        'sensor': sensor,
        'filepath': filepath
    }
    
    return raw, metadata


def prepare_features(
    raw_data: np.ndarray,
    patch_size: int,
    n_channels: int,
    pm_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convert raw data into model inputs (patch, context) and labels.
    
    Args:
        raw_data: Array of samples from load_training_data
        patch_size: Spatial dimension of patch
        n_channels: Number of spectral channels
        pm_threshold: Threshold for binary classification (µg/L)
        
    Returns:
        Tuple of (X_patch, X_context, y_binary, dates, context_size)
    """
    # Filter to only samples with the requested patch size
    # Sample format: (filename, station, labels_tuple, features, valid_fraction, patch_size)
    filtered_data = [s for s in raw_data if s[5] == patch_size]
    if len(filtered_data) == 0:
        raise ValueError(f"No samples found with patch_size={patch_size}")
    
    logging.info(f"Filtered to {len(filtered_data)} samples with patch_size={patch_size} "
                 f"(from {len(raw_data)} total)")
    
    raw_data = np.array(filtered_data, dtype=object)
    
    # Determine feature dimensions
    first_sample = raw_data[0]
    first_flat = first_sample[3]
    n_features = first_flat.size
    
    feat_patch = patch_size * patch_size * n_channels
    context_size = n_features - feat_patch
    
    if context_size <= 0:
        raise ValueError(
            f"Invalid context size: {context_size}. "
            f"Features={n_features}, patch_features={feat_patch}"
        )
    
    logging.info(f"Feature dimensions: patch={feat_patch}, context={context_size}")
    
    # Build arrays AND extract dates for temporal splitting
    n = len(raw_data)
    X_all = np.zeros((n, n_features), dtype=float)
    y_all = np.zeros((n,), dtype=float)
    dates = []
    
    for i, sample in enumerate(raw_data):
        # sample format: (filename, station, labels, flat_features, ...)
        filename, _, labels, flat_features, *_ = sample
        X_all[i] = flat_features
        
        # Extract particulate microcystin concentration
        # labels[4] is PM concentration in µg/L
        particulate_mc = labels[4] if len(labels) >= 5 and not np.isnan(labels[4]) else 0.01
        y_all[i] = particulate_mc
        
        # Extract date from filename (format: PACE_OCI.20240517T160942.L2.OC.V2_0.NRT.nc)
        try:
            date_str = filename.split('.')[1][:8]  # YYYYMMDD
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            dates.append(date_obj)
        except Exception as e:
            logging.warning(f"Could not parse date from {filename}: {e}")
            dates.append(datetime(2024, 1, 1))  # Fallback date
    
    dates = np.array(dates)
    
    # Split patch and context features
    X_patch_flat = X_all[:, :feat_patch]
    X_context = X_all[:, feat_patch:]
    
    # Reshape patch to 4D: (n_samples, patch_size, patch_size, n_channels)
    X_patch = X_patch_flat.reshape((n, patch_size, patch_size, n_channels))
    
    # Add binary mask channel (1 where data exists, 0 for NaN)
    mask = np.any(~np.isnan(X_patch), axis=-1, keepdims=True).astype('float32')
    X_patch = np.concatenate([X_patch, mask], axis=-1)
    
    # Create binary labels
    y_binary = (y_all >= pm_threshold).astype('float32')
    
    logging.info(f"Class distribution: positive={y_binary.sum():.0f}, "
                 f"negative={len(y_binary) - y_binary.sum():.0f}")
    
    return X_patch, X_context, y_binary, dates, context_size


def normalize_features(
    X_patch: np.ndarray,
    X_context: np.ndarray,
    n_channels: int,
    save_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize patch and context features, save statistics.
    
    Args:
        X_patch: Patch array (n, patch_size, patch_size, n_channels + 1)
        X_context: Context array (n, context_size)
        n_channels: Number of spectral channels (excluding mask)
        save_dir: Directory to save normalization stats
        
    Returns:
        Tuple of (X_patch_norm, X_context_norm)
    """
    # Normalize patch (excluding mask channel)
    patch_means = np.nanmean(X_patch[..., :n_channels], axis=(0, 1, 2))
    patch_stds = np.nanstd(X_patch[..., :n_channels], axis=(0, 1, 2))
    
    X_patch[..., :n_channels] = (
        (X_patch[..., :n_channels] - patch_means) / (patch_stds + 1e-6)
    )
    X_patch = np.nan_to_num(X_patch, nan=0.0)
    
    # Normalize context
    context_means = np.nanmean(X_context, axis=0)
    context_stds = np.nanstd(X_context, axis=0)
    
    X_context = (X_context - context_means) / (context_stds + 1e-6)
    X_context = np.nan_to_num(X_context, nan=0.0)
    
    # Save normalization statistics
    os.makedirs(os.path.join(save_dir, 'channel_stats'), exist_ok=True)
    np.save(os.path.join(save_dir, 'channel_stats', 'means.npy'), patch_means)
    np.save(os.path.join(save_dir, 'channel_stats', 'stds.npy'), patch_stds)
    np.save(os.path.join(save_dir, 'context_means.npy'), context_means)
    np.save(os.path.join(save_dir, 'context_stds.npy'), context_stds)
    
    logging.info(f"Saved normalization stats to {save_dir}")
    
    return X_patch, X_context


def augment_data(
    X_patch: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply data augmentation via spatial flips.
    
    Args:
        X_patch: Patch array (n, patch_size, patch_size, n_channels + 1)
        X_context: Context array (n, context_size)
        y: Labels (n,)
        patch_size: Spatial dimension of patch
        
    Returns:
        Tuple of (X_patch_aug, X_context_aug, y_aug) with 4× more samples
    """
    if patch_size <= 1:
        logging.warning("Patch size <= 1, skipping augmentation")
        return X_patch, X_context, y
    
    # Create flipped versions: horizontal, vertical, both
    flips = [
        np.flip(X_patch, axis=2),  # Horizontal flip
        np.flip(X_patch, axis=1),  # Vertical flip
        np.flip(np.flip(X_patch, axis=1), axis=2)  # Both
    ]
    
    # Concatenate original + 3 flipped versions
    X_patch_aug = np.concatenate([X_patch] + flips, axis=0)
    X_context_aug = np.concatenate([X_context] * 4, axis=0)
    y_aug = np.concatenate([y] * 4, axis=0)
    
    logging.info(f"Augmented data: {len(X_patch)} → {len(X_patch_aug)} samples")
    
    return X_patch_aug, X_context_aug, y_aug


def create_train_val_test_split(
    X_patch: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.125,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train/val/test sets using TEMPORAL grouping.
    
    CRITICAL FIX FOR DATA LEAKAGE:
    - Groups samples by date to prevent temporal leakage
    - Ensures samples from same day stay together in one split
    - Prevents model from "seeing" test dates during training
    
    Split strategy:
    - 20% test (by dates)
    - 12.5% validation (by dates, from remaining 80%)
    - 67.5% train
    
    Args:
        X_patch: Patch features
        X_context: Context features
        y: Labels
        dates: Date array (one per sample)
        test_size: Fraction for test set
        val_size: Fraction of remaining data for validation  
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_patch_train, X_patch_val, X_patch_test,
                  X_context_train, X_context_val, X_context_test,
                  y_train, y_val, y_test)
    """
    # Create date groups (format: YYYYMMDD as string)
    date_groups = np.array([d.strftime('%Y%m%d') for d in dates])
    unique_dates = np.unique(date_groups)
    
    logging.info(f"Temporal split: {len(unique_dates)} unique dates, {len(dates)} total samples")
    logging.info(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    
    # First split: separate test set BY DATE
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X_patch, y, groups=date_groups))
    
    X_patch_tmp = X_patch[train_val_idx]
    X_context_tmp = X_context[train_val_idx]
    y_tmp = y[train_val_idx]
    date_groups_tmp = date_groups[train_val_idx]
    
    X_patch_test = X_patch[test_idx]
    X_context_test = X_context[test_idx]
    y_test = y[test_idx]
    
    # Second split: validation from remaining BY DATE
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_patch_tmp, y_tmp, groups=date_groups_tmp))
    
    X_patch_train = X_patch_tmp[train_idx]
    X_context_train = X_context_tmp[train_idx]
    y_train = y_tmp[train_idx]
    
    X_patch_val = X_patch_tmp[val_idx]
    X_context_val = X_context_tmp[val_idx]
    y_val = y_tmp[val_idx]
    
    # Report split statistics
    train_dates = np.unique(date_groups[train_val_idx][train_idx])
    val_dates = np.unique(date_groups[train_val_idx][val_idx])
    test_dates = np.unique(date_groups[test_idx])
    
    logging.info(f"\nSplit sizes (samples):")
    logging.info(f"  Train: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    logging.info(f"  Val:   {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    logging.info(f"  Test:  {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")
    
    logging.info(f"\nSplit sizes (unique dates):")
    logging.info(f"  Train: {len(train_dates)} dates")
    logging.info(f"  Val:   {len(val_dates)} dates")
    logging.info(f"  Test:  {len(test_dates)} dates")
    
    logging.info(f"\nClass distribution:")
    logging.info(f"  Train: {y_train.sum():.0f} pos / {len(y_train)-y_train.sum():.0f} neg")
    logging.info(f"  Val:   {y_val.sum():.0f} pos / {len(y_val)-y_val.sum():.0f} neg")
    logging.info(f"  Test:  {y_test.sum():.0f} pos / {len(y_test)-y_test.sum():.0f} neg")
    
    # CRITICAL CHECK: Ensure no date overlap
    date_overlap_train_val = set(train_dates) & set(val_dates)
    date_overlap_train_test = set(train_dates) & set(test_dates)
    date_overlap_val_test = set(val_dates) & set(test_dates)
    
    if date_overlap_train_val or date_overlap_train_test or date_overlap_val_test:
        logging.error("❌ DATE LEAKAGE DETECTED!")
        logging.error(f"  Train-Val overlap: {len(date_overlap_train_val)} dates")
        logging.error(f"  Train-Test overlap: {len(date_overlap_train_test)} dates")
        logging.error(f"  Val-Test overlap: {len(date_overlap_val_test)} dates")
        raise ValueError("Date leakage detected in splits!")
    else:
        logging.info("✅ No date overlap between splits - temporal leakage prevented")
    
    return (X_patch_train, X_patch_val, X_patch_test,
            X_context_train, X_context_val, X_context_test,
            y_train, y_val, y_test)


def train_model(
    save_dir: str = './',
    sensor: str = 'PACE',
    patch_size: int = 3,
    pm_threshold: float = 0.1,
    epochs: int = 300,
    batch_size: int = 64,
    early_stopping: bool = True
) -> Tuple[float, float, float, float]:
    """
    Complete training pipeline for microcystin detection CNN.
    
    Args:
        save_dir: Directory for saving model and stats
        sensor: Sensor name ('PACE' or 'Sentinel-3')
        patch_size: Spatial dimension of patch (3, 5, 7, or 9)
        pm_threshold: Threshold for binary classification (µg/L)
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        early_stopping: Whether to use early stopping
        
    Returns:
        Tuple of (loss, accuracy, auc, f1_score) on test set
    """
    # Get sensor parameters
    sensor_params = config.SENSOR_PARAMS[sensor]
    n_channels = len(sensor_params['channels'])
    
    logging.info(f"Training {sensor} model: patch_size={patch_size}, "
                 f"channels={n_channels}, threshold={pm_threshold}")
    
    # ===== LOAD DATA =====
    raw_data, metadata = load_training_data(save_dir, sensor)
    
    # ===== PREPARE FEATURES (NOW INCLUDES DATES) =====
    X_patch, X_context, y_binary, dates, context_size = prepare_features(
        raw_data, patch_size, n_channels, pm_threshold
    )
    
    # ===== NORMALIZE =====
    X_patch, X_context = normalize_features(
        X_patch, X_context, n_channels, save_dir
    )
    
    # ===== SPLIT FIRST (CRITICAL FIX #1: Split before augmentation) =====
    logging.info("\n" + "="*70)
    logging.info("CRITICAL FIX: Splitting BEFORE augmentation to prevent leakage")
    logging.info("="*70)
    
    splits = create_train_val_test_split(X_patch, X_context, y_binary, dates)
    (X_patch_train, X_patch_val, X_patch_test,
     X_context_train, X_context_val, X_context_test,
     y_train, y_val, y_test) = splits
    
    # ===== AUGMENT ONLY TRAINING SET (CRITICAL FIX #2: No augmentation of val/test) =====
    logging.info("\n" + "="*70)
    logging.info("CRITICAL FIX: Augmenting ONLY training set (not val/test)")
    logging.info("="*70)
    
    X_patch_train, X_context_train, y_train = augment_data(
        X_patch_train, X_context_train, y_train, patch_size
    )
    
    logging.info(f"Final dataset sizes:")
    logging.info(f"  Train: {len(y_train)} samples (augmented)")
    logging.info(f"  Val:   {len(y_val)} samples (original, no augmentation)")
    logging.info(f"  Test:  {len(y_test)} samples (original, no augmentation)")
    
    # ===== BUILD MODEL =====
    model = build_model(
        patch_size=patch_size,
        n_channels=n_channels,
        context_size=context_size,
        learning_rate=config.LEARNING_RATE
    )
    
    # ===== CALLBACKS =====
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            )
        )
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        )
    
    # ===== TRAIN =====
    logging.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        [X_patch_train, X_context_train],
        y_train,
        validation_data=([X_patch_val, X_context_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks
    )
    
    # ===== EVALUATE =====
    logging.info("Evaluating on test set...")
    loss, acc, auc, precision, recall = model.evaluate(
        [X_patch_test, X_context_test],
        y_test,
        verbose=2
    )
    
    # Calculate F1 score
    y_pred_prob = model.predict([X_patch_test, X_context_test], verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    # Print detailed metrics
    logging.info(f"\nTest Results:")
    logging.info(f"  Loss:      {loss:.4f}")
    logging.info(f"  Accuracy:  {acc:.4f}")
    logging.info(f"  AUC:       {auc:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")
    
    # Classification report
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logging.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.npy')
    np.save(history_path, history.history)
    logging.info(f"Saved training history to {history_path}")
    
    return loss, acc, auc, f1


if __name__ == '__main__':
    import argparse
    from . import config
    
    parser = argparse.ArgumentParser(description='Train microcystin detection CNN')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing training data (default: from config)')
    parser.add_argument('--sensor', type=str, default='PACE',
                        choices=['PACE', 'Sentinel-3'],
                        help='Sensor type')
    parser.add_argument('--patch-size', type=int, default=3,
                        choices=[3, 5, 7, 9],
                        help='Patch size (pixels per side)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='PM threshold for binary classification (µg/L)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping')
    
    args = parser.parse_args()
    
    # Use config directory if not specified
    data_dir = args.data_dir if args.data_dir is not None else str(config.BASE_DIR)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Train model
    loss, acc, auc, f1 = train_model(
        save_dir=data_dir,
        sensor=args.sensor,
        patch_size=args.patch_size,
        pm_threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=not args.no_early_stopping
    )
    
    logging.info(f"\nFinal test metrics: loss={loss:.4f}, acc={acc:.4f}, "
                 f"auc={auc:.4f}, f1={f1:.4f}")
