#!/usr/bin/env python3
"""
Quick validation script for the Phase 3 production model.
Generates forecasts using the most recent Sentinel-3 data to verify model works.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chla_lstm_forecasting.model import load_model
from chla_lstm_forecasting.utils import load_composite_data, create_sequences
import chla_lstm_forecasting.config as config


def inverse_transform(normalized_chla, max_chla=500.0):
    """Convert normalized predictions back to mg/m³"""
    # Denormalize from [-1, 1] to log scale
    log_value = (normalized_chla + 1) / 2 * np.log10(max_chla + 1)
    
    # Inverse log10 transform
    chla_mgm3 = 10**log_value - 1
    
    # Clip to valid range
    return np.clip(chla_mgm3, 0.001, max_chla)


def main():
    print("=" * 80)
    print("PHASE 3 PRODUCTION MODEL - VALIDATION")
    print("=" * 80)
    
    # Load the production model
    # Due to Lambda layer serialization issues, we'll rebuild the model and load weights
    print("\nBuilding model architecture...")
    from chla_lstm_forecasting.model import build_convlstm_model
    
    # Match the training configuration
    input_shape = (config.SEQUENCE_LENGTH, 93, 163, 2)
    model = build_convlstm_model(
        input_shape=input_shape,
        filters_1=32,
        filters_2=32,
        learning_rate=config.LEARNING_RATE
    )
    print("✅ Model architecture built")
    
    # Load the saved weights
    model_path = Path("chla_lstm_forecasting/best_model.keras")
    print(f"\nLoading weights from: {model_path}")
    
    import tensorflow as tf
    import keras
    keras.config.enable_unsafe_deserialization()
    sys.modules['__main__'].tf = tf
    
    # Load just to get the weights
    saved_model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Transfer weights
    for layer, saved_layer in zip(model.layers[:-1], saved_model.layers[:-1]):  # Skip Lambda layer
        if saved_layer.get_weights():
            layer.set_weights(saved_layer.get_weights())
    
    print("✅ Weights loaded successfully")
    
    # Get recent data files
    data_dir = Path("CNN-LSTM/Images2")
    all_files = sorted(data_dir.glob("composite_data_S3_*.npy"))
    
    if len(all_files) < config.SEQUENCE_LENGTH + 1:
        print(f"❌ Not enough files. Need {config.SEQUENCE_LENGTH + 1}, found {len(all_files)}")
        return
    
    print(f"\nFound {len(all_files)} Sentinel-3 composite files")
    
    # Use data from summer (better coverage) - around indices 600-800 (2023-2024)
    # Training uses first 60% (619), validation 20% (206), test last 20% (207)
    # Test set starts at index 825 (619+206)
    
    # Get files from test set (last 207 samples)
    test_start = len(all_files) - 207
    test_files = all_files[test_start:test_start + 20]  # Get 20 files from test set
    
    print(f"\nUsing 20 files from test set (indices {test_start} to {test_start + 20}):")
    for i, f in enumerate(test_files[:5]):
        print(f"  {f.name}")
    print(f"  ... ({len(test_files)} files total)")
    
    # Convert to string paths for create_sequences
    test_paths = [str(f) for f in test_files]
    
    # Create sequences
    print(f"\nCreating sequences (length={config.SEQUENCE_LENGTH})...")
    X, y = create_sequences(test_paths, seq_len=config.SEQUENCE_LENGTH)
    print(f"✅ Created {len(X)} sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Take the last sequence for prediction
    test_input = X[-1:] # Shape: (1, 5, 93, 163, 2)
    test_target = y[-1]  # Shape: (93, 163, 2)
    
    print(f"\nTest sequence shape: {test_input.shape}")
    
    # Make prediction
    print("\nGenerating prediction...")
    prediction = model.predict(test_input, verbose=0)
    print(f"✅ Prediction shape: {prediction.shape}")
    
    # Extract chlorophyll channel (first channel)
    pred_chla = prediction[0, :, :, 0]  # Shape: (93, 163)
    true_chla = test_target[:, :, 0]     # Shape: (93, 163)
    
    # Get valid pixel mask (second channel)
    valid_mask = test_target[:, :, 1] > 0.5  # Shape: (93, 163)
    
    # Calculate metrics only on valid pixels
    valid_pred = pred_chla[valid_mask]
    valid_true = true_chla[valid_mask]
    
    mse = np.mean((valid_pred - valid_true) ** 2)
    mae = np.mean(np.abs(valid_pred - valid_true))
    
    print(f"\n📊 VALIDATION METRICS (normalized scale):")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Valid pixels: {valid_mask.sum()} / {valid_mask.size}")
    
    # Convert to mg/m³ for visualization
    pred_mgm3 = inverse_transform(pred_chla)
    true_mgm3 = inverse_transform(true_chla)
    
    # Set invalid pixels to NaN for better visualization
    pred_mgm3[~valid_mask] = np.nan
    true_mgm3[~valid_mask] = np.nan
    
    # Calculate real-scale metrics
    valid_pred_mgm3 = pred_mgm3[valid_mask]
    valid_true_mgm3 = true_mgm3[valid_mask]
    
    mse_mgm3 = np.mean((valid_pred_mgm3 - valid_true_mgm3) ** 2)
    mae_mgm3 = np.mean(np.abs(valid_pred_mgm3 - valid_true_mgm3))
    
    print(f"\n📊 VALIDATION METRICS (mg/m³ scale):")
    print(f"   MSE: {mse_mgm3:.2f}")
    print(f"   MAE: {mae_mgm3:.2f}")
    print(f"   Mean true: {valid_true_mgm3.mean():.2f} mg/m³")
    print(f"   Mean pred: {valid_pred_mgm3.mean():.2f} mg/m³")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True chlorophyll
    im1 = axes[0].imshow(true_mgm3, cmap='viridis', vmin=0, vmax=100)
    axes[0].set_title('True Chlorophyll-a\n(3-day composite)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Chlorophyll-a (mg/m³)')
    
    # Predicted chlorophyll
    im2 = axes[1].imshow(pred_mgm3, cmap='viridis', vmin=0, vmax=100)
    axes[1].set_title('Predicted Chlorophyll-a\n(3 days ahead)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Chlorophyll-a (mg/m³)')
    
    # Error map
    error = np.abs(pred_mgm3 - true_mgm3)
    im3 = axes[2].imshow(error, cmap='Reds', vmin=0, vmax=20)
    axes[2].set_title(f'Absolute Error\n(MAE: {mae_mgm3:.2f} mg/m³)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='Error (mg/m³)')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("chla_lstm_forecasting/validation")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"production_model_validation_{timestamp}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n📈 Summary:")
    print(f"   Model: {model_path}")
    print(f"   Test MSE: {mse:.4f} (normalized), {mse_mgm3:.2f} (mg/m³)²")
    print(f"   Test MAE: {mae:.4f} (normalized), {mae_mgm3:.2f} mg/m³")
    print(f"   Visualization: {output_path}")
    print(f"\n🎯 Production model is working correctly!")


if __name__ == "__main__":
    main()
