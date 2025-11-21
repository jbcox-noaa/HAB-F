"""
Utility functions for chlorophyll-a forecasting.

This module handles:
- Data loading and preprocessing
- Sequence creation for temporal modeling
- Normalization and masking
- File I/O operations
"""

import os
import logging
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from . import config


def configure_logging():
    """Configure logging for the module."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_plot(
    fig: plt.Figure,
    basename: str = 'plot',
    suffix: Optional[str] = None,
    fmt: str = 'png',
    dpi: int = 150,
    output_dir: Optional[str] = None
) -> str:
    """
    Save a matplotlib figure with timestamp.
    
    Args:
        fig: Matplotlib figure to save
        basename: Base name for the file
        suffix: Optional suffix before timestamp
        fmt: Image format (png, pdf, etc.)
        dpi: Resolution in dots per inch
        output_dir: Output directory (default: config.PLOTS_DIR)
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [basename]
    if suffix:
        parts.append(suffix)
    parts.append(timestamp)
    fname = '_'.join(parts) + f'.{fmt}'
    fpath = os.path.join(output_dir, fname)
    
    fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot to {fpath}")
    
    return fpath


def parse_file(
    data_path: str,
    band_index: int = None,
    max_chla: float = None
) -> np.ndarray:
    """
    Parse and preprocess a chlorophyll data file.
    
    This function:
    1. Loads raw data from .npy file
    2. Extracts the specified band (default: chlorophyll)
    3. Handles invalid/NaN values
    4. Applies log transformation
    5. Normalizes to [-1, 1]
    6. Creates a validity mask
    
    Args:
        data_path: Path to .npy data file
        band_index: Index of band to extract (default from config)
        max_chla: Maximum chlorophyll value for clipping (default from config)
        
    Returns:
        Array of shape (H, W, 2) where:
            channel 0 = normalized log-chlorophyll [-1, 1]
            channel 1 = validity mask [0 or 1]
    """
    if band_index is None:
        band_index = config.CHLA_BAND_INDEX
    if max_chla is None:
        max_chla = config.MAX_CHLA
    
    # Load data
    data = np.load(data_path)
    
    # Drop any leading time axis if present
    if data.ndim == 4:
        data = data[0]
    
    # Extract chlorophyll band
    raw = data[..., band_index]
    
    # Create validity mask
    valid = np.isfinite(raw)
    
    # Replace invalid values with small positive constant
    safe = np.where(valid, raw, 0.001)
    clamped = np.clip(safe, 0.001, max_chla)
    
    # Log10 transformation (matching original LSTM.py)
    # Add 1 to avoid log10(0) and match original preprocessing
    log_band = np.log10(clamped + 1)
    
    # Normalize by global constant (not per-file max)
    # This ensures consistent scaling across all files
    norm_factor = np.log10(max_chla + 1)  # log10(501) ≈ 2.7
    scaled = (log_band / norm_factor) * 2 - 1
    
    # Remove any lingering NaNs and enforce mask
    scaled = np.nan_to_num(scaled, nan=-1.0)
    scaled = np.where(valid, scaled, -1.0)
    
    # Build mask channel
    mask = valid.astype(np.float32)
    
    # Stack channels
    combined = np.stack([scaled, mask], axis=-1)
    
    return combined.astype(np.float32)


def create_sequences(
    data_files: List[str],
    seq_len: int = 5,
    band_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create temporal sequences for training.
    
    Loads a series of data files and creates overlapping sequences
    for supervised learning. Each sequence contains `seq_len` time steps
    as input (X) and the next time step as target (y).
    
    Args:
        data_files: List of paths to .npy data files (sorted chronologically)
        seq_len: Number of time steps in each input sequence
        band_index: Index of band to extract (default from config)
        
    Returns:
        Tuple of (X, y) where:
            X: shape (num_samples, seq_len, H, W, 2)
            y: shape (num_samples, H, W, 2)
    """
    # Load and parse all frames
    frames = []
    for d_path in data_files:
        arr = parse_file(d_path, band_index)
        # arr shape: (H, W, 2) where channels are [scaled, mask]
        frames.append(arr)
    
    # Stack into (N, H, W, 2)
    frames = np.stack(frames, axis=0)
    
    # Create sequences
    num_samples = len(frames) - seq_len
    if num_samples <= 0:
        raise ValueError(f"Not enough frames ({len(frames)}) to create sequences of length {seq_len}")
    
    X_list, y_list = [], []
    for i in range(num_samples):
        # Input: seq_len consecutive frames
        X_list.append(frames[i:i+seq_len])    # (seq_len, H, W, 2)
        # Target: next frame
        y_list.append(frames[i+seq_len])      # (H, W, 2)
    
    X = np.array(X_list, dtype=np.float32)    # (num_samples, seq_len, H, W, 2)
    y = np.array(y_list, dtype=np.float32)    # (num_samples, H, W, 2)
    
    logging.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
    logging.info(f"  Number of samples: {num_samples}")
    logging.info(f"  Sequence length: {seq_len}")
    logging.info(f"  Spatial dimensions: {X.shape[2]}×{X.shape[3]}")
    
    return X, y


def split_temporal_data(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets temporally.
    
    Important: This splits chronologically to prevent data leakage.
    Do NOT shuffle before splitting!
    
    Args:
        X: Input sequences (num_samples, seq_len, H, W, 2)
        y: Target frames (num_samples, H, W, 2)
        train_frac: Fraction for training (default 0.6)
        val_frac: Fraction for validation (default 0.2)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    total = len(X)
    train_end = int(train_frac * total)
    val_end = int((train_frac + val_frac) * total)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]
    
    logging.info(f"Split data temporally:")
    logging.info(f"  Train: {len(X_train)} samples ({100*train_frac:.1f}%)")
    logging.info(f"  Val:   {len(X_val)} samples ({100*val_frac:.1f}%)")
    logging.info(f"  Test:  {len(X_test)} samples ({100*(1-train_frac-val_frac):.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_chlorophyll_map(
    data: np.ndarray,
    title: str,
    lons: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = 'viridis',
    coastline_res: str = '10m',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot chlorophyll-a data on a map.
    
    Args:
        data: 2D array of chlorophyll values
        title: Plot title
        lons: Longitude coordinates (optional)
        lats: Latitude coordinates (optional)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        coastline_res: Coastline resolution ('10m', '50m', '110m')
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Add map features
    ax.coastlines(resolution=coastline_res)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Plot data
    if lons is not None and lats is not None:
        im = ax.pcolormesh(
            lons, lats, data,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax,
            cmap=cmap
        )
    else:
        im = ax.pcolormesh(
            data,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax,
            cmap=cmap
        )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Chlorophyll-a', fontsize=10)
    
    # Title
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Grid
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    plt.tight_layout()
    
    return fig


def load_composite_data(
    directory: str,
    sensor: str = "S3",
    limit: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Load composite data and metadata file paths from directory.
    
    Args:
        directory: Path to data directory
        sensor: Sensor identifier ("S3" for Sentinel-3, "PACE" for PACE)
        limit: Maximum number of files to load (None for all)
        
    Returns:
        Tuple of (data_files, metadata_files) sorted chronologically
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    # Find data files
    data_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(f'{config.COMPOSITE_DATA_PREFIX}_{sensor}_')
        and f.endswith('.npy')
    ])
    
    # Find metadata files
    meta_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(f'{config.COMPOSITE_METADATA_PREFIX}_{sensor}_')
        and f.endswith('.npy')
    ])
    
    if not data_files:
        raise ValueError(f"No data files found for sensor {sensor} in {directory}")
    
    # Apply limit if specified
    if limit is not None:
        data_files = data_files[:limit]
        meta_files = meta_files[:limit] if meta_files else []
    
    logging.info(f"Loaded {len(data_files)} data files for sensor {sensor}")
    if meta_files:
        logging.info(f"Loaded {len(meta_files)} metadata files")
    
    return data_files, meta_files


def validate_data(X: np.ndarray, y: np.ndarray):
    """
    Validate data arrays for training.
    
    Checks for NaN, Inf, and prints statistics.
    
    Args:
        X: Input sequences
        y: Target frames
    """
    logging.info("Data validation:")
    logging.info(f"  X shape: {X.shape}, dtype: {X.dtype}")
    logging.info(f"  y shape: {y.shape}, dtype: {y.dtype}")
    logging.info(f"  X range: [{np.nanmin(X):.3f}, {np.nanmax(X):.3f}]")
    logging.info(f"  y range: [{np.nanmin(y):.3f}, {np.nanmax(y):.3f}]")
    logging.info(f"  X NaNs: {np.isnan(X).sum()}, Infs: {np.isinf(X).sum()}")
    logging.info(f"  y NaNs: {np.isnan(y).sum()}, Infs: {np.isinf(y).sum()}")
    
    if np.isnan(X).any() or np.isinf(X).any():
        logging.warning("X contains NaN or Inf values!")
    if np.isnan(y).any() or np.isinf(y).any():
        logging.warning("y contains NaN or Inf values!")
