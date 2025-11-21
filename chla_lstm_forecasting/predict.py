"""
Prediction and forecasting utilities for chlorophyll-a forecasting model.

This module handles:
- Single-step predictions
- Multi-step autoregressive forecasting
- Visualization of predictions
- Time series forecasting
"""

import os
import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from . import config
from .utils import (
    configure_logging,
    parse_file,
    plot_chlorophyll_map,
    save_plot
)
from .model import load_model


def denormalize_chlorophyll(
    normalized: np.ndarray,
    max_val: float = None
) -> np.ndarray:
    """
    Convert normalized [-1, 1] values back to chlorophyll concentrations.
    
    Args:
        normalized: Normalized array (values in [-1, 1])
        max_val: Maximum chlorophyll value used in normalization
        
    Returns:
        Denormalized chlorophyll concentrations (mg/m³)
    """
    if max_val is None:
        max_val = config.MAX_CHLA
    
    # Convert from [-1, 1] to [0, 1]
    norm_01 = (normalized + 1.0) / 2.0
    
    # Apply inverse log transformation
    # log_val = log10(chla + 1), so chla = 10^log_val - 1
    log_val = norm_01 * np.log10(max_val + 1)
    chla = np.power(10, log_val) - 1
    
    return chla


def predict_single_step(
    model: keras.Model,
    sequence: np.ndarray
) -> np.ndarray:
    """
    Predict next timestep from input sequence.
    
    Args:
        model: Trained ConvLSTM model
        sequence: Input sequence (seq_len, H, W, channels)
        
    Returns:
        Predicted frame (H, W, channels)
    """
    # Add batch dimension
    X = np.expand_dims(sequence, axis=0)  # (1, seq_len, H, W, channels)
    
    # Predict
    y_pred = model.predict(X, verbose=0)  # (1, H, W, 1)
    
    # Remove batch dimension
    pred = y_pred[0]  # (H, W, 1)
    
    return pred


def predict_multi_step(
    model: keras.Model,
    initial_sequence: np.ndarray,
    n_steps: int = 7
) -> List[np.ndarray]:
    """
    Autoregressive multi-step forecasting.
    
    Args:
        model: Trained ConvLSTM model
        initial_sequence: Initial sequence (seq_len, H, W, channels)
        n_steps: Number of steps to forecast
        
    Returns:
        List of predicted frames
    """
    predictions = []
    sequence = initial_sequence.copy()
    
    for step in range(n_steps):
        # Predict next frame
        pred = predict_single_step(model, sequence)
        predictions.append(pred)
        
        # Update sequence: drop oldest frame, append new prediction
        # sequence shape: (seq_len, H, W, channels)
        # pred shape: (H, W, 1)
        
        # If sequence has 2 channels (data + mask), extract just the mask
        if sequence.shape[-1] == 2:
            mask_channel = sequence[-1, :, :, 1:2]  # Most recent mask
            pred_with_mask = np.concatenate([pred, mask_channel], axis=-1)
        else:
            pred_with_mask = pred
        
        # Roll sequence: remove first frame, append prediction
        sequence = np.concatenate([
            sequence[1:],  # Drop oldest
            np.expand_dims(pred_with_mask, axis=0)  # Add newest
        ], axis=0)
    
    return predictions


def visualize_forecast(
    predictions: List[np.ndarray],
    dates: Optional[List[str]] = None,
    output_path: Optional[str] = None
):
    """
    Visualize multi-step forecast as a series of maps.
    
    Args:
        predictions: List of predicted frames (H, W, 1)
        dates: Optional list of date strings for titles
        output_path: Path to save figure
    """
    n_pred = len(predictions)
    
    # Create subplots
    ncols = min(4, n_pred)
    nrows = (n_pred + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    if n_pred == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    for i, pred in enumerate(predictions):
        ax = axes[i]
        
        # Extract chlorophyll channel (first channel)
        chla_norm = pred[:, :, 0]
        
        # Denormalize
        chla = denormalize_chlorophyll(chla_norm)
        
        # Plot
        im = ax.imshow(chla, cmap='YlGn', vmin=0, vmax=50)
        
        # Title
        if dates and i < len(dates):
            title = f"Forecast: {dates[i]}"
        else:
            title = f"Step {i+1}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Chl-a (mg/m³)', fraction=0.046)
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_pred, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=config.PLOT_DPI)
        logging.info(f"Saved forecast visualization to {output_path}")
    else:
        save_plot(fig, basename='forecast')
    
    plt.close()


def plot_time_series_at_point(
    predictions: List[np.ndarray],
    row: int,
    col: int,
    dates: Optional[List[str]] = None,
    output_path: Optional[str] = None
):
    """
    Plot forecasted chlorophyll time series at a specific pixel.
    
    Args:
        predictions: List of predicted frames
        row: Row index
        col: Column index
        dates: Optional list of date strings
        output_path: Path to save figure
    """
    # Extract chlorophyll values at point
    chla_values = []
    for pred in predictions:
        chla_norm = pred[row, col, 0]
        chla = denormalize_chlorophyll(chla_norm)
        chla_values.append(chla)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(chla_values))
    ax.plot(x, chla_values, marker='o', linewidth=2, markersize=8)
    
    # Labels
    ax.set_xlabel('Forecast Step', fontsize=12)
    ax.set_ylabel('Chl-a (mg/m³)', fontsize=12)
    ax.set_title(f'Forecasted Chlorophyll at Pixel ({row}, {col})', 
                 fontsize=14, fontweight='bold')
    
    # X-axis labels
    if dates:
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=config.PLOT_DPI)
        logging.info(f"Saved time series plot to {output_path}")
    else:
        save_plot(fig, basename='time_series')
    
    plt.close()


def run_prediction(
    model_path: str,
    data_files: List[str],
    seq_len: int = None,
    n_steps: int = 1,
    output_dir: Optional[str] = None
):
    """
    Run prediction pipeline.
    
    Args:
        model_path: Path to trained model
        data_files: List of input data files (chronologically sorted)
        seq_len: Sequence length (default from config)
        n_steps: Number of steps to forecast
        output_dir: Output directory for visualizations
    """
    configure_logging()
    
    if seq_len is None:
        seq_len = config.SEQUENCE_LENGTH
    if output_dir is None:
        output_dir = str(config.PLOTS_DIR)
    
    logging.info("=" * 70)
    logging.info("CHLOROPHYLL-A FORECASTING - PREDICTION")
    logging.info("=" * 70)
    logging.info(f"Model: {model_path}")
    logging.info(f"Input files: {len(data_files)}")
    logging.info(f"Sequence length: {seq_len}")
    logging.info(f"Forecast steps: {n_steps}")
    
    # ===== LOAD MODEL =====
    logging.info("\nLoading model...")
    model = load_model(model_path)
    
    # ===== PREPARE INPUT SEQUENCE =====
    logging.info("\nPreparing input sequence...")
    
    if len(data_files) < seq_len:
        raise ValueError(
            f"Not enough files for sequence: {len(data_files)} "
            f"(need {seq_len})"
        )
    
    # Use most recent files
    recent_files = data_files[-seq_len:]
    
    # Parse files into sequence
    frames = []
    for fpath in recent_files:
        frame = parse_file(fpath)  # (H, W, 2)
        frames.append(frame)
        logging.info(f"  Loaded: {os.path.basename(fpath)}")
    
    sequence = np.array(frames)  # (seq_len, H, W, 2)
    logging.info(f"Input sequence shape: {sequence.shape}")
    
    # ===== RUN PREDICTION =====
    logging.info(f"\nRunning {n_steps}-step forecast...")
    
    if n_steps == 1:
        pred = predict_single_step(model, sequence)
        predictions = [pred]
    else:
        predictions = predict_multi_step(model, sequence, n_steps=n_steps)
    
    logging.info(f"Generated {len(predictions)} predictions")
    
    # ===== VISUALIZE =====
    logging.info("\nGenerating visualizations...")
    
    # Forecast maps
    forecast_path = os.path.join(output_dir, f'forecast_{n_steps}step.png')
    visualize_forecast(predictions, output_path=forecast_path)
    
    # Time series at center pixel
    H, W = predictions[0].shape[:2]
    center_row, center_col = H // 2, W // 2
    ts_path = os.path.join(output_dir, f'timeseries_center.png')
    plot_time_series_at_point(
        predictions,
        center_row,
        center_col,
        output_path=ts_path
    )
    
    # ===== SUMMARY =====
    logging.info("\n" + "=" * 70)
    logging.info("PREDICTION COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"Visualizations saved to: {output_dir}")
    logging.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run chlorophyll-a forecasting predictions'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.keras file)'
    )
    
    parser.add_argument(
        '--data-files',
        type=str,
        nargs='+',
        required=True,
        help='Input data files (chronologically sorted)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=None,
        help=f'Sequence length (default: {config.SEQUENCE_LENGTH})'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=1,
        help='Number of forecast steps (default: 1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: {config.PLOTS_DIR})'
    )
    
    args = parser.parse_args()
    
    run_prediction(
        model_path=args.model,
        data_files=args.data_files,
        seq_len=args.seq_len,
        n_steps=args.n_steps,
        output_dir=args.output_dir
    )
