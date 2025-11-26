"""
Utility functions for microcystin probability forecasting.

This module handles:
- Loading MC probability maps (.npy files)
- Creating temporal sequences with gap handling
- Temporal train/val/test split (2024 vs 2025)
- Data validation and quality checks
"""

import os
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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


def load_mc_map(filepath: str) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load a single MC probability map and its metadata.
    
    Args:
        filepath: Path to .npy probability map file
        
    Returns:
        Tuple of (probability_map, metadata_dict)
        probability_map: shape (H, W) with values in [0, 1]
        metadata_dict: Contains lat/lon coordinates if available
    """
    # Load probability map
    prob_map = np.load(filepath)
    
    # Try to load corresponding coordinate file
    coords_path = filepath.replace('.npy', '_coords.npz')
    metadata = None
    
    if os.path.exists(coords_path):
        coords = np.load(coords_path)
        metadata = {
            'lats': coords['lats'],
            'lons': coords['lons']
        }
    
    return prob_map, metadata


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from MC probability map filename.
    
    Expected format: mc_probability_YYYYMMDD.npy
    
    Args:
        filename: Name of the probability map file
        
    Returns:
        datetime object or None if parsing fails
    """
    try:
        # Extract date string (8 digits)
        date_str = filename.replace('mc_probability_', '').replace('.npy', '')
        return datetime.strptime(date_str, '%Y%m%d')
    except (ValueError, IndexError):
        return None


def load_all_mc_maps(
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[List[np.ndarray], List[datetime], Dict]:
    """
    Load all MC probability maps from directory, sorted chronologically.
    
    Args:
        data_dir: Directory containing mc_probability_*.npy files
        start_date: Optional start date filter (YYYYMMDD)
        end_date: Optional end date filter (YYYYMMDD)
        
    Returns:
        Tuple of (maps, dates, metadata) where:
            maps: List of probability arrays, each shape (H, W)
            dates: List of corresponding datetime objects
            metadata: Dict with lat/lon coordinates (from first map)
    """
    data_path = Path(data_dir)
    map_files = sorted(data_path.glob('mc_probability_*.npy'))
    
    if len(map_files) == 0:
        raise FileNotFoundError(f"No MC probability maps found in {data_dir}")
    
    logging.info(f"Found {len(map_files)} MC probability map files")
    
    # Parse dates and filter
    file_date_pairs = []
    for f in map_files:
        date = parse_date_from_filename(f.name)
        if date is None:
            logging.warning(f"Could not parse date from {f.name}, skipping")
            continue
        
        # Apply date filters if provided
        if start_date and date < datetime.strptime(start_date, '%Y%m%d'):
            continue
        if end_date and date > datetime.strptime(end_date, '%Y%m%d'):
            continue
            
        file_date_pairs.append((f, date))
    
    # Sort chronologically
    file_date_pairs.sort(key=lambda x: x[1])
    
    logging.info(f"Loading {len(file_date_pairs)} maps after date filtering")
    logging.info(f"  Date range: {file_date_pairs[0][1].strftime('%Y-%m-%d')} to "
                f"{file_date_pairs[-1][1].strftime('%Y-%m-%d')}")
    
    # Load all maps
    maps = []
    dates = []
    metadata = None
    
    for filepath, date in file_date_pairs:
        prob_map, meta = load_mc_map(str(filepath))
        maps.append(prob_map)
        dates.append(date)
        
        # Store metadata from first map
        if metadata is None and meta is not None:
            metadata = meta
    
    logging.info(f"Loaded {len(maps)} MC probability maps")
    logging.info(f"  Map shape: {maps[0].shape}")
    logging.info(f"  Value range: [{np.nanmin(maps[0]):.3f}, {np.nanmax(maps[0]):.3f}]")
    
    return maps, dates, metadata


def create_sequences_with_gap_handling(
    maps: List[np.ndarray],
    dates: List[datetime],
    seq_len: int = 5,
    forecast_horizon: int = 1,
    max_gap_days: int = 3
) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    """
    Create temporal sequences from MC probability maps with gap handling.
    
    This function creates sequences only when:
    1. All required dates in lookback window exist
    2. Gaps between consecutive dates are <= max_gap_days
    
    This prevents temporal discontinuities while accepting small gaps
    (e.g., 2-3 day gaps due to clouds) that are common in satellite data.
    
    Args:
        maps: List of probability arrays, each shape (H, W)
        dates: List of corresponding datetime objects (sorted)
        seq_len: Number of days in lookback window (default: 5)
        forecast_horizon: Days ahead to predict (default: 1)
        max_gap_days: Maximum allowed gap between dates in sequence (default: 3)
        
    Returns:
        Tuple of (X, y, target_dates) where:
            X: Input sequences, shape (N, seq_len, H, W, 1)
            y: Target maps, shape (N, H, W, 1)
            target_dates: Dates corresponding to each target (length N)
            
    Notes:
        - One map per day (no temporal compositing to avoid leakage)
        - Maps already aggregate multiple PACE overpasses per day
        - Sequences with gaps > max_gap_days are skipped
        - See docs/PHASE5_TEMPORAL_DATA_ANALYSIS.md for rationale
    """
    X_list = []
    y_list = []
    target_dates = []
    
    skipped_gaps = 0
    skipped_horizon = 0
    
    # Need seq_len + forecast_horizon dates total
    for i in range(len(dates) - seq_len - forecast_horizon + 1):
        # Get dates for this sequence
        seq_dates = dates[i:i+seq_len]
        target_idx = i + seq_len + forecast_horizon - 1
        
        # Check if target date exists
        if target_idx >= len(dates):
            skipped_horizon += 1
            continue
        
        target_date = dates[target_idx]
        
        # Check gaps within sequence
        gaps = [(seq_dates[j] - seq_dates[j-1]).days 
                for j in range(1, len(seq_dates))]
        
        if max(gaps) > max_gap_days:
            skipped_gaps += 1
            continue
        
        # Check gap to target
        gap_to_target = (target_date - seq_dates[-1]).days
        if gap_to_target > max_gap_days:
            skipped_gaps += 1
            continue
        
        # Valid sequence - extract maps
        seq_maps = [maps[i+j] for j in range(seq_len)]
        target_map = maps[target_idx]
        
        # Stack to arrays with channel dimension
        X_seq = np.stack(seq_maps, axis=0)  # (seq_len, H, W)
        X_seq = X_seq[..., np.newaxis]      # (seq_len, H, W, 1)
        
        y_map = target_map[..., np.newaxis]  # (H, W, 1)
        
        X_list.append(X_seq)
        y_list.append(y_map)
        target_dates.append(target_date)
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    logging.info(f"Created {len(X)} sequences from {len(maps)} maps")
    logging.info(f"  Sequence length: {seq_len} days")
    logging.info(f"  Forecast horizon: {forecast_horizon} days")
    logging.info(f"  Max gap allowed: {max_gap_days} days")
    logging.info(f"  Skipped due to gaps: {skipped_gaps}")
    logging.info(f"  Skipped due to horizon: {skipped_horizon}")
    logging.info(f"  X shape: {X.shape}")
    logging.info(f"  y shape: {y.shape}")
    
    if len(X) == 0:
        raise ValueError(f"No valid sequences created. Try increasing max_gap_days or "
                        f"decreasing seq_len/forecast_horizon")
    
    return X, y, target_dates


def split_by_year(
    X: np.ndarray,
    y: np.ndarray,
    dates: List[datetime],
    train_year: str = "2024",
    val_year: str = "2025",
    test_year: str = "2025",
    val_end_date: str = "20250801",
    test_start_date: str = "20250801"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences by temporal period to prevent data leakage.
    
    Temporal split strategy:
    - Train: All data from train_year (2024)
    - Val: Data from val_year before val_end_date (2025 Jan-Jul, early bloom)
    - Test: Data from test_year after test_start_date (2025 Aug-Oct, peak bloom)
    
    This ensures:
    1. No data leakage (training data all precedes validation/test)
    2. Both val and test contain bloom season data (user requirement)
    3. Val has early bloom, test has peak bloom (different dynamics)
    
    Args:
        X: Input sequences (N, seq_len, H, W, 1)
        y: Target maps (N, H, W, 1)
        dates: List of target dates (length N)
        train_year: Year for training data (default: "2024")
        val_year: Year for validation data (default: "2025")
        test_year: Year for test data (default: "2025")
        val_end_date: End date for validation, YYYYMMDD (default: "20250801")
        test_start_date: Start date for test, YYYYMMDD (default: "20250801")
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        
    See Also:
        docs/PHASE5_TEMPORAL_SPLIT_RATIONALE.md for detailed explanation
    """
    dates_array = np.array(dates)
    
    # Parse split dates
    val_end = datetime.strptime(val_end_date, '%Y%m%d')
    test_start = datetime.strptime(test_start_date, '%Y%m%d')
    
    # Create masks
    train_mask = np.array([d.year == int(train_year) for d in dates])
    val_mask = np.array([
        d.year == int(val_year) and d < val_end 
        for d in dates
    ])
    test_mask = np.array([
        d.year == int(test_year) and d >= test_start 
        for d in dates
    ])
    
    # Verify no overlap
    assert not np.any(train_mask & val_mask), "Train and val overlap!"
    assert not np.any(train_mask & test_mask), "Train and test overlap!"
    assert not np.any(val_mask & test_mask), "Val and test overlap!"
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Log split info
    logging.info(f"Temporal split by year:")
    logging.info(f"  Train ({train_year}): {len(X_train)} sequences")
    if len(X_train) > 0:
        train_dates = dates_array[train_mask]
        logging.info(f"    Date range: {train_dates[0].strftime('%Y-%m-%d')} to "
                    f"{train_dates[-1].strftime('%Y-%m-%d')}")
    
    logging.info(f"  Val ({val_year} < {val_end_date}): {len(X_val)} sequences")
    if len(X_val) > 0:
        val_dates = dates_array[val_mask]
        logging.info(f"    Date range: {val_dates[0].strftime('%Y-%m-%d')} to "
                    f"{val_dates[-1].strftime('%Y-%m-%d')}")
        logging.info(f"    Contains early bloom season")
    
    logging.info(f"  Test ({test_year} >= {test_start_date}): {len(X_test)} sequences")
    if len(X_test) > 0:
        test_dates = dates_array[test_mask]
        logging.info(f"    Date range: {test_dates[0].strftime('%Y-%m-%d')} to "
                    f"{test_dates[-1].strftime('%Y-%m-%d')}")
        logging.info(f"    Contains peak bloom season")
    else:
        test_dates = np.array([])
    
    # Extract date arrays for each split
    train_dates_list = dates_array[train_mask] if len(X_train) > 0 else np.array([])
    val_dates_list = dates_array[val_mask] if len(X_val) > 0 else np.array([])
    test_dates_list = dates_array[test_mask] if len(X_test) > 0 else np.array([])
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_dates_list, val_dates_list, test_dates_list)


def load_mc_sequences(
    data_dir: str = None,
    seq_len: int = None,
    forecast_horizon: int = None,
    max_gap_days: int = None,
    train_year: str = None,
    val_end_date: str = None,
    test_start_date: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Complete pipeline to load MC probability maps and create train/val/test splits.
    
    This is the main function to use for data loading. It:
    1. Loads all MC probability maps from data directory
    2. Creates temporal sequences with gap handling
    3. Splits into train/val/test by year
    4. Returns ready-to-use data for model training
    
    Args:
        data_dir: Directory with MC probability maps (default: config.DATA_DIR)
        seq_len: Sequence length in days (default: config.SEQ_LEN)
        forecast_horizon: Days ahead to predict (default: config.FORECAST_HORIZON)
        max_gap_days: Max gap in sequence (default: config.MAX_GAP_DAYS)
        train_year: Training year (default: config.TRAIN_YEAR)
        val_end_date: Validation end date (default: config.VAL_END_DATE)
        test_start_date: Test start date (default: config.TEST_START_DATE)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        
    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
        >>> print(f"Training on {len(X_train)} sequences")
        >>> print(f"Map shape: {X_train.shape[2:4]}")  # (H, W)
    """
    # Use config defaults if not specified
    data_dir = data_dir or config.DATA_DIR
    seq_len = seq_len or config.SEQ_LEN
    forecast_horizon = forecast_horizon or config.FORECAST_HORIZON
    max_gap_days = max_gap_days or config.MAX_GAP_DAYS
    train_year = train_year or config.TRAIN_YEAR
    val_end_date = val_end_date or config.VAL_END_DATE
    test_start_date = test_start_date or config.TEST_START_DATE
    
    logging.info("=" * 80)
    logging.info("LOADING MC PROBABILITY SEQUENCES")
    logging.info("=" * 80)
    
    # Load all maps
    maps, dates, metadata = load_all_mc_maps(data_dir)
    
    # Create sequences
    X, y, target_dates = create_sequences_with_gap_handling(
        maps=maps,
        dates=dates,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        max_gap_days=max_gap_days
    )
    
    # Split by year
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_dates, val_dates, test_dates) = split_by_year(
        X=X,
        y=y,
        dates=target_dates,
        train_year=train_year,
        val_end_date=val_end_date,
        test_start_date=test_start_date
    )
    
    # Add dates to metadata
    metadata['train_dates'] = [d.strftime('%Y%m%d') for d in train_dates]
    metadata['val_dates'] = [d.strftime('%Y%m%d') for d in val_dates]
    metadata['test_dates'] = [d.strftime('%Y%m%d') for d in test_dates]
    
    logging.info("=" * 80)
    logging.info("DATA LOADING COMPLETE")
    logging.info("=" * 80)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def plot_mc_probability_map(
    data: np.ndarray,
    title: str,
    lons: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = 'YlOrRd',
    coastline_res: str = '10m',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot MC probability data on a map.
    
    Args:
        data: Probability map, shape (H, W) or (H, W, 1)
        title: Plot title
        lons: Longitude grid (optional)
        lats: Latitude grid (optional)
        vmin: Minimum value for colorbar (default: 0.0)
        vmax: Maximum value for colorbar (default: 1.0)
        cmap: Colormap name (default: 'YlOrRd' for probabilities)
        coastline_res: Cartopy coastline resolution
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Remove channel dimension if present
    if data.ndim == 3:
        data = data[..., 0]
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent (Lake Erie region)
    ax.set_extent([-83.6, -78.8, 41.3, 42.9], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale(coastline_res), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(coastline_res), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale(coastline_res), linewidth=0.3)
    ax.add_feature(cfeature.LAKES.with_scale(coastline_res), alpha=0.5)
    
    # Plot data
    if lons is not None and lats is not None:
        im = ax.pcolormesh(lons, lats, data, 
                          vmin=vmin, vmax=vmax, cmap=cmap,
                          transform=ccrs.PlateCarree())
    else:
        im = ax.imshow(data, extent=[-83.6, -78.8, 41.3, 42.9],
                      vmin=vmin, vmax=vmax, cmap=cmap,
                      origin='lower', transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Microcystin Probability', fontsize=10)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return fig


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
