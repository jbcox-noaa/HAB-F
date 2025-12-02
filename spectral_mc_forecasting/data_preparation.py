"""
Data Preparation for Spectral MC Forecasting (Phase 7)

This module handles:
1. Loading raw PACE granules (172-band spectral data)
2. Loading GLERL ground truth microcystin measurements
3. Creating spatial grid aligned with MC probability maps
4. Matching GLERL point measurements to PACE spectra
5. Processing daily spectral maps
6. Creating temporal sequences for unsupervised pre-training
7. Computing normalization statistics
8. Saving matched labeled data and unlabeled sequences

Output:
- glerl_pace_matched.npz: Labeled samples (GLERL-PACE pairs) for supervised training
- spectral_sequences_{train,val,test}.h5: Unlabeled spectral sequences for pre-training
- normalization_stats_train.npz: Per-band mean and std
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict
import logging
from tqdm import tqdm
import h5py

from spectral_mc_forecasting import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_glerl_measurements(glerl_csv_path: str = '/Users/jessecox/Desktop/NOAA/HAB-F/GLERL_GT/glrl-hab-data.csv') -> pd.DataFrame:
    """
    Load GLERL microcystin measurements (all depth categories).
    
    Includes Surface, Bottom, Mid-column, and Scum measurements.
    
    Args:
        glerl_csv_path: Path to GLERL CSV file
        
    Returns:
        DataFrame with columns: timestamp, station_name, lat, lon, particulate_microcystin, extracted_chla
    """
    logger.info(f"Loading GLERL measurements from {glerl_csv_path}")
    
    df = pd.read_csv(glerl_csv_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Keep only needed columns
    df = df[['timestamp', 'lat', 'lon', 'particulate_microcystin']].copy()
    
    # Create date string for matching
    df['date_str'] = df['timestamp'].dt.strftime('%Y%m%d')
    
    # Create binary label (MC >= 1.0 µg/L)
    df['mc_binary'] = (df['particulate_microcystin'] >= config.MC_THRESHOLD).astype(int)
    
    logger.info(f"Loaded {len(df)} GLERL measurements")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"  Positive samples (MC >= {config.MC_THRESHOLD}): {df['mc_binary'].sum()} ({100*df['mc_binary'].mean():.1f}%)")
    
    return df


def match_glerl_to_pace_spectra(
    glerl_df: pd.DataFrame,
    daily_maps: Dict[str, np.ndarray],
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    max_spatial_dist_km: float = 1.2,  # One pixel length
    temporal_window_days: int = 1  # ±1 day window
) -> List[Dict]:
    """
    Match GLERL point measurements to PACE spectral data.
    
    Args:
        glerl_df: DataFrame with GLERL measurements
        daily_maps: Dictionary of date_str -> (H, W, 172) spectral maps
        grid_lats: (H,) grid latitudes
        grid_lons: (W,) grid longitudes
        max_spatial_dist_km: Maximum distance for spatial matching (1.2km = 1 pixel)
        temporal_window_days: Days before/after GLERL measurement to search for PACE data
        
    Returns:
        List of matched samples with structure:
        {
            'glerl_date_str': original GLERL measurement date,
            'pace_date_str': matched PACE granule date,
            'temporal_offset_days': days between GLERL and PACE,
            'lat': measurement latitude,
            'lon': measurement longitude,
            'grid_row': matched grid row,
            'grid_col': matched grid column,
            'spectra': (172,) spectral signature,
            'mc_value': microcystin concentration,
            'mc_binary': binary label (0 or 1)
        }
    """
    from scipy.spatial import cKDTree
    from datetime import timedelta
    
    logger.info("="*80)
    logger.info("MATCHING GLERL MEASUREMENTS TO PACE SPECTRA")
    logger.info("="*80)
    logger.info(f"Max spatial distance: {max_spatial_dist_km} km")
    logger.info(f"Temporal window: ±{temporal_window_days} days")
    
    # Create grid mesh for spatial matching
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
    grid_points = np.column_stack([grid_lat_2d.ravel(), grid_lon_2d.ravel()])
    
    matched_samples = []
    unmatched_temporal = []
    unmatched_spatial = []
    
    for _, row in tqdm(glerl_df.iterrows(), total=len(glerl_df), desc="Matching GLERL to PACE"):
        glerl_date_str = row['date_str']
        glerl_date = datetime.strptime(glerl_date_str, '%Y%m%d')
        glerl_lat = row['lat']
        glerl_lon = row['lon']
        
        # Search for PACE data within temporal window
        best_match = None
        best_temporal_offset = None
        
        for day_offset in range(-temporal_window_days, temporal_window_days + 1):
            search_date = glerl_date + timedelta(days=day_offset)
            search_date_str = search_date.strftime('%Y%m%d')
            
            if search_date_str not in daily_maps:
                continue
            
            spectral_map = daily_maps[search_date_str]  # (H, W, 172)
            
            # Find nearest grid cell using KD-tree
            glerl_point = np.array([[glerl_lat, glerl_lon]])
            tree = cKDTree(grid_points)
            distance, idx = tree.query(glerl_point, k=1)
            distance = distance[0]
            idx = idx[0]
            
            # Convert distance from degrees to km (approximate)
            distance_km = distance * 111.0  # 1 degree ~ 111 km
            
            if distance_km > max_spatial_dist_km:
                continue  # Try next day in window
            
            # Get grid indices
            grid_row = idx // len(grid_lons)
            grid_col = idx % len(grid_lons)
            
            # Extract spectral signature
            spectra = spectral_map[grid_row, grid_col, :]  # (172,)
            
            # Check if spectral data is valid (not all NaN)
            if np.all(np.isnan(spectra)):
                continue  # Try next day in window
            
            # Found a valid match - keep the closest temporal match
            if best_match is None or abs(day_offset) < abs(best_temporal_offset):
                best_match = {
                    'glerl_date_str': glerl_date_str,
                    'pace_date_str': search_date_str,
                    'temporal_offset_days': day_offset,
                    'timestamp': row['timestamp'],
                    'lat': glerl_lat,
                    'lon': glerl_lon,
                    'grid_row': grid_row,
                    'grid_col': grid_col,
                    'spectra': spectra,
                    'mc_value': row['particulate_microcystin'],
                    'mc_binary': row['mc_binary'],
                    'spatial_distance_km': distance_km
                }
                best_temporal_offset = day_offset
        
        if best_match is not None:
            matched_samples.append(best_match)
        elif glerl_date_str not in [d for d in daily_maps.keys()]:
            unmatched_temporal.append(glerl_date_str)
        else:
            unmatched_spatial.append((glerl_date_str, 'no_valid_match_in_window'))
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MATCHING RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total GLERL measurements: {len(glerl_df)}")
    logger.info(f"Successfully matched: {len(matched_samples)}")
    logger.info(f"Unmatched (no PACE data in window): {len(unmatched_temporal)}")
    logger.info(f"Unmatched (spatial/quality): {len(unmatched_spatial)}")
    
    if len(matched_samples) > 0:
        mc_binary_counts = pd.Series([s['mc_binary'] for s in matched_samples]).value_counts()
        temporal_offsets = pd.Series([s['temporal_offset_days'] for s in matched_samples])
        spatial_distances = pd.Series([s['spatial_distance_km'] for s in matched_samples])
        
        logger.info(f"\nMatched sample distribution:")
        logger.info(f"  Negative (MC < {config.MC_THRESHOLD}): {mc_binary_counts.get(0, 0)} ({100*mc_binary_counts.get(0, 0)/len(matched_samples):.1f}%)")
        logger.info(f"  Positive (MC >= {config.MC_THRESHOLD}): {mc_binary_counts.get(1, 0)} ({100*mc_binary_counts.get(1, 0)/len(matched_samples):.1f}%)")
        logger.info(f"\nTemporal offsets:")
        logger.info(f"  Same day (0): {(temporal_offsets == 0).sum()}")
        logger.info(f"  ±1 day: {(temporal_offsets.abs() == 1).sum()}")
        logger.info(f"  Mean offset: {temporal_offsets.abs().mean():.2f} days")
        logger.info(f"\nSpatial distances:")
        logger.info(f"  Mean: {spatial_distances.mean():.3f} km")
        logger.info(f"  Max: {spatial_distances.max():.3f} km")
    
    return matched_samples


def load_pace_spectra(granule_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load full spectral data from PACE granule.
    
    Args:
        granule_path: Path to PACE L2 AOP granule (.nc file)
        
    Returns:
        Tuple of (spectra, wavelengths, lats, lons) or (None, None, None, None) if failed
        - spectra: (n_pixels, 172) array of reflectance values
        - wavelengths: (172,) array of wavelengths in nm
        - lats: (n_pixels,) array of latitudes
        - lons: (n_pixels,) array of longitudes
    """
    try:
        # Open granule
        with xr.open_dataset(granule_path, group='navigation_data') as nav:
            lats = nav['latitude'].values
            lons = nav['longitude'].values
        
        # Load wavelengths
        with xr.open_dataset(granule_path, group='sensor_band_parameters') as sensor:
            wavelengths = sensor['wavelength_3d'].values  # Shape: (172,)
        
        # Load remote sensing reflectance (Rrs)
        with xr.open_dataset(granule_path, group='geophysical_data') as geo:
            # Rrs is a 3D array: (n_scan, n_pixel, n_bands)
            spectra = geo['Rrs'].values  # Already (n_scan, n_pixel, 172)
        
        # Flatten spatial dimensions
        spectra_flat = spectra.reshape(-1, config.N_SPECTRAL_BANDS)  # (n_pixels, 172)
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        
        logger.debug(f"Loaded {granule_path.split('/')[-1]}: {spectra_flat.shape[0]:,} pixels")
        
        return spectra_flat, wavelengths, lats_flat, lons_flat
        
    except Exception as e:
        logger.error(f"Failed to load {granule_path}: {e}")
        return None, None, None, None


def filter_to_bbox(spectra: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data to Lake Erie bounding box.
    
    Args:
        spectra: (n_pixels, 172) spectral data
        lats: (n_pixels,) latitudes
        lons: (n_pixels,) longitudes
        
    Returns:
        Filtered (spectra, lats, lons)
    """
    bbox = config.BBOX
    mask = (
        (lons >= bbox[0]) & (lons <= bbox[2]) &
        (lats >= bbox[1]) & (lats <= bbox[3])
    )
    
    return spectra[mask], lats[mask], lons[mask]


def grid_spectra_to_map(
    spectra: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray
) -> np.ndarray:
    """
    Grid spectral data onto regular spatial grid using nearest-neighbor.
    
    Args:
        spectra: (n_pixels, 172) spectral data
        lats: (n_pixels,) pixel latitudes
        lons: (n_pixels,) pixel longitudes
        grid_lats: (H,) grid latitude centers
        grid_lons: (W,) grid longitude centers
        
    Returns:
        Gridded spectra: (H, W, 172) with NaN for empty cells
    """
    from scipy.spatial import cKDTree
    
    H, W = len(grid_lats), len(grid_lons)
    gridded = np.full((H, W, config.N_SPECTRAL_BANDS), np.nan, dtype=np.float32)
    
    # Create grid mesh
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
    grid_points = np.column_stack([grid_lat_2d.ravel(), grid_lon_2d.ravel()])
    
    # Build KD-tree for data points
    data_points = np.column_stack([lats, lons])
    tree = cKDTree(data_points)
    
    # For each grid cell, find nearest data point
    distances, indices = tree.query(grid_points, k=1, distance_upper_bound=0.05)  # ~5km threshold
    
    # Fill grid
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if not np.isinf(dist):  # Valid nearest neighbor
            row = i // W
            col = i % W
            gridded[row, col, :] = spectra[idx, :]
    
    return gridded


def create_spatial_grid() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spatial grid aligned with MC probability maps.
    
    Returns:
        (grid_lats, grid_lons) of shape (H,) and (W,)
    """
    bbox = config.BBOX
    
    # Create grid centers
    grid_lons = np.linspace(bbox[0], bbox[2], config.WIDTH)
    grid_lats = np.linspace(bbox[1], bbox[3], config.HEIGHT)
    
    return grid_lats, grid_lons


def process_pace_granule_to_map(granule_path: str, grid_lats: np.ndarray, grid_lons: np.ndarray) -> Optional[np.ndarray]:
    """
    Process single PACE granule to gridded spectral map.
    
    Args:
        granule_path: Path to PACE granule
        grid_lats: (H,) grid latitudes
        grid_lons: (W,) grid longitudes
        
    Returns:
        Spectral map (H, W, 172) or None if failed
    """
    # Load spectra
    spectra, wavelengths, lats, lons = load_pace_spectra(granule_path)
    if spectra is None:
        return None
    
    # Filter to bbox
    spectra, lats, lons = filter_to_bbox(spectra, lats, lons)
    
    if len(spectra) == 0:
        logger.warning(f"No pixels in bbox for {Path(granule_path).name}")
        return None
    
    # Grid to map
    spectral_map = grid_spectra_to_map(spectra, lats, lons, grid_lats, grid_lons)
    
    # Check coverage
    valid_pct = 100 * (~np.isnan(spectral_map[:, :, 0])).sum() / (config.HEIGHT * config.WIDTH)
    logger.debug(f"{Path(granule_path).name}: {valid_pct:.1f}% coverage")
    
    return spectral_map


def extract_date_from_granule(granule_path: str) -> Optional[str]:
    """
    Extract date string from PACE granule filename.
    
    Example: PACE_OCI.20240506T180930.L2.OC_AOP.V3_1.nc -> 20240506
    
    Returns:
        Date string (YYYYMMDD) or None
    """
    try:
        filename = Path(granule_path).name
        parts = filename.split('.')
        datetime_str = parts[1]  # 20240506T180930
        date_str = datetime_str.split('T')[0]  # 20240506
        return date_str
    except Exception as e:
        logger.error(f"Failed to extract date from {granule_path}: {e}")
        return None


def aggregate_daily_maps(date_maps: List[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple spectral maps for same day.
    
    Args:
        date_maps: List of (H, W, 172) arrays
        
    Returns:
        Aggregated map (H, W, 172)
    """
    if len(date_maps) == 1:
        return date_maps[0]
    
    # Stack and take mean (ignoring NaN)
    stacked = np.stack(date_maps, axis=0)  # (n_maps, H, W, 172)
    aggregated = np.nanmean(stacked, axis=0)  # (H, W, 172)
    
    return aggregated


def process_all_pace_granules() -> Dict[str, np.ndarray]:
    """
    Process all PACE granules to daily spectral maps.
    
    Returns:
        Dictionary mapping date_str (YYYYMMDD) to spectral map (H, W, 172)
    """
    logger.info("="*80)
    logger.info("PROCESSING PACE GRANULES TO SPECTRAL MAPS")
    logger.info("="*80)
    
    # Create spatial grid
    grid_lats, grid_lons = create_spatial_grid()
    logger.info(f"Spatial grid: {len(grid_lats)}x{len(grid_lons)} ({config.HEIGHT}x{config.WIDTH})")
    logger.info(f"Grid bounds: lat [{grid_lats[0]:.3f}, {grid_lats[-1]:.3f}], lon [{grid_lons[0]:.3f}, {grid_lons[-1]:.3f}]")
    
    # Find all PACE granules
    granule_files = sorted(config.DATA_DIR.glob("PACE_OCI*.nc"))
    logger.info(f"Found {len(granule_files)} PACE granules")
    
    # Group by date
    date_groups = {}
    for granule_path in granule_files:
        date_str = extract_date_from_granule(str(granule_path))
        if date_str is None:
            continue
        
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(str(granule_path))
    
    logger.info(f"Granules span {len(date_groups)} unique dates")
    
    # Process each date
    daily_maps = {}
    failed_dates = []
    
    for date_str in tqdm(sorted(date_groups.keys()), desc="Processing dates"):
        granule_paths = date_groups[date_str]
        
        # Process all granules for this date
        date_maps = []
        for granule_path in granule_paths:
            spectral_map = process_pace_granule_to_map(granule_path, grid_lats, grid_lons)
            if spectral_map is not None:
                date_maps.append(spectral_map)
        
        if len(date_maps) == 0:
            logger.warning(f"No valid maps for {date_str}")
            failed_dates.append(date_str)
            continue
        
        # Aggregate if multiple granules per day
        if len(date_maps) > 1:
            logger.debug(f"{date_str}: Aggregating {len(date_maps)} granules")
        
        daily_maps[date_str] = aggregate_daily_maps(date_maps)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {len(daily_maps)} dates")
    logger.info(f"Failed: {len(failed_dates)} dates")
    logger.info(f"Date range: {min(daily_maps.keys())} to {max(daily_maps.keys())}")
    
    return daily_maps


def extract_patch(
    spectral_map: np.ndarray,
    row: int,
    col: int,
    patch_size: int = config.PATCH_SIZE,
    use_mask_channel: bool = config.USE_MASK_CHANNEL
) -> np.ndarray:
    """
    Extract a spatial patch centered on a grid cell.
    
    Handles edge cases where patch extends beyond map boundaries by zero-padding.
    
    Args:
        spectral_map: (H, W, 172) or (H, W, 173) spectral map
        row: Center row index
        col: Center column index
        patch_size: Size of patch (must be odd, e.g., 7, 11, 15)
        use_mask_channel: If True, expect/return 173 channels (172 + mask)
        
    Returns:
        Extracted patch of shape (patch_size, patch_size, 172 or 173)
        Zero-padded if patch extends beyond boundaries
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    
    H, W = spectral_map.shape[:2]
    n_channels = spectral_map.shape[2]
    half = patch_size // 2
    
    # Initialize zero-filled patch
    patch = np.zeros((patch_size, patch_size, n_channels), dtype=np.float32)
    
    # Calculate source region in spectral_map
    src_row_start = max(0, row - half)
    src_row_end = min(H, row + half + 1)
    src_col_start = max(0, col - half)
    src_col_end = min(W, col + half + 1)
    
    # Calculate destination region in patch
    dst_row_start = max(0, half - row)
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = max(0, half - col)
    dst_col_end = dst_col_start + (src_col_end - src_col_start)
    
    # Copy data
    patch[dst_row_start:dst_row_end, dst_col_start:dst_col_end, :] = \
        spectral_map[src_row_start:src_row_end, src_col_start:src_col_end, :]
    
    return patch


def create_patch_based_sequences(
    matched_samples: List[Dict],
    daily_maps: Dict[str, np.ndarray],
    seq_len: int = config.SEQ_LEN,
    patch_size: int = config.PATCH_SIZE,
    min_valid_days: int = config.MIN_VALID_DAYS,
    use_mask_channel: bool = config.USE_MASK_CHANNEL
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Create patch-based temporal sequences for GLERL-labeled samples.
    
    For each GLERL measurement:
    1. Find its matched PACE date and grid location
    2. Extract spatial patch around that location
    3. Build 14-day lookback sequence of patches
    4. Use center pixel as ground truth label
    
    Args:
        matched_samples: List of GLERL-PACE matched samples from match_glerl_to_pace_spectra()
        daily_maps: Dictionary mapping date_str to (H, W, 172) spectral maps
        seq_len: Lookback window length (default 14 days)
        patch_size: Size of spatial patch (default 11x11)
        min_valid_days: Minimum number of valid days required in sequence
        use_mask_channel: If True, add mask channel (1=valid, 0=missing)
        
    Returns:
        (X_sequences, y_labels, metadata)
        - X_sequences: (N, seq_len, patch_size, patch_size, 173) input sequences
        - y_labels: (N,) binary labels (0 or 1)
        - metadata: List of dicts with sample info (date, lat, lon, MC value, etc.)
    """
    logger.info("="*80)
    logger.info("CREATING PATCH-BASED SEQUENCES FOR GLERL SAMPLES")
    logger.info("="*80)
    logger.info(f"Matched samples: {len(matched_samples)}")
    logger.info(f"Lookback window: {seq_len} days")
    logger.info(f"Patch size: {patch_size}x{patch_size} (~{patch_size * config.RES_KM:.1f}km x {patch_size * config.RES_KM:.1f}km)")
    logger.info(f"Min valid days required: {min_valid_days}")
    logger.info(f"Use mask channel: {use_mask_channel}")
    
    X_sequences = []
    y_labels = []
    metadata = []
    
    n_channels = config.N_INPUT_FEATURES if use_mask_channel else config.N_SPECTRAL_BANDS
    
    skipped_insufficient_history = 0
    
    for sample in tqdm(matched_samples, desc="Creating patch sequences"):
        pace_date_str = sample['pace_date_str']
        pace_date = datetime.strptime(pace_date_str, '%Y%m%d')
        grid_row = sample['grid_row']
        grid_col = sample['grid_col']
        mc_binary = sample['mc_binary']
        
        # Define the lookback window: [pace_date - seq_len, ..., pace_date - 1]
        lookback_dates = []
        for offset in range(seq_len, 0, -1):
            lookback_date = pace_date - timedelta(days=offset)
            lookback_dates.append(lookback_date.strftime('%Y%m%d'))
        
        # Check which dates have valid data
        valid_mask = np.array([d in daily_maps for d in lookback_dates], dtype=np.float32)
        valid_count = int(valid_mask.sum())
        
        # Skip if insufficient valid days
        if valid_count < min_valid_days:
            skipped_insufficient_history += 1
            continue
        
        # Build the patch sequence
        sequence = []
        for j, date_str in enumerate(lookback_dates):
            if valid_mask[j] == 1:
                # Valid day - extract patch from spectral map
                spectral_map = daily_maps[date_str]  # (H, W, 172)
                
                # Add mask channel
                if use_mask_channel:
                    mask = np.ones((config.HEIGHT, config.WIDTH, 1), dtype=np.float32)
                    spectral_map_with_mask = np.concatenate([spectral_map, mask], axis=-1)  # (H, W, 173)
                else:
                    spectral_map_with_mask = spectral_map
                
                # Extract patch
                patch = extract_patch(spectral_map_with_mask, grid_row, grid_col, patch_size, use_mask_channel)
                
            else:
                # Missing day - use zeros with mask=0
                patch = np.zeros((patch_size, patch_size, n_channels), dtype=np.float32)
                if use_mask_channel:
                    # Keep all-zero (mask channel is 0 indicating missing)
                    pass
            
            sequence.append(patch)
        
        # Stack into array
        X_seq = np.stack(sequence, axis=0)  # (seq_len, patch_size, patch_size, 173 or 172)
        
        # Store
        X_sequences.append(X_seq)
        y_labels.append(mc_binary)
        
        # Metadata
        meta = {
            'pace_date': pace_date_str,
            'glerl_date': sample['glerl_date_str'],
            'temporal_offset_days': sample['temporal_offset_days'],
            'lat': sample['lat'],
            'lon': sample['lon'],
            'grid_row': grid_row,
            'grid_col': grid_col,
            'mc_value': sample['mc_value'],
            'mc_binary': mc_binary,
            'lookback_dates': lookback_dates,
            'valid_count': valid_count,
            'sparsity': 1.0 - (valid_count / seq_len)
        }
        metadata.append(meta)
    
    # Convert to arrays
    X_sequences = np.array(X_sequences, dtype=np.float32)  # (N, seq_len, patch_size, patch_size, 173)
    y_labels = np.array(y_labels, dtype=np.int32)  # (N,)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PATCH-BASED SEQUENCES CREATED")
    logger.info(f"{'='*80}")
    logger.info(f"Total sequences: {len(X_sequences)}")
    logger.info(f"Skipped (insufficient history): {skipped_insufficient_history}")
    logger.info(f"Sequence shape: {X_sequences.shape}")
    logger.info(f"Labels shape: {y_labels.shape}")
    logger.info(f"\nClass distribution:")
    logger.info(f"  Negative (MC < {config.MC_THRESHOLD}): {(y_labels == 0).sum()} ({100*(y_labels == 0).mean():.1f}%)")
    logger.info(f"  Positive (MC >= {config.MC_THRESHOLD}): {(y_labels == 1).sum()} ({100*(y_labels == 1).mean():.1f}%)")
    
    if len(metadata) > 0:
        sparsities = [m['sparsity'] for m in metadata]
        valid_counts = [m['valid_count'] for m in metadata]
        logger.info(f"\nSequence statistics:")
        logger.info(f"  Mean valid days: {np.mean(valid_counts):.1f} / {seq_len}")
        logger.info(f"  Mean sparsity: {np.mean(sparsities):.2%}")
        logger.info(f"  Dense sequences (all valid): {sum(1 for s in sparsities if s == 0)} ({100*sum(1 for s in sparsities if s == 0)/len(sparsities):.1f}%)")
    
    return X_sequences, y_labels, metadata


def create_temporal_sequences(
    daily_maps: Dict[str, np.ndarray],
    seq_len: int = config.SEQ_LEN,
    min_valid_days: int = config.MIN_VALID_DAYS,
    use_mask_channel: bool = config.USE_MASK_CHANNEL
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[Dict]]:
    """
    Create temporal sequences from daily spectral maps with support for missing data.
    
    Uses a fixed lookback window and allows sparse sequences (missing days are masked).
    
    Args:
        daily_maps: Dictionary mapping date_str to (H, W, 172) spectral maps
        seq_len: Lookback window length (default 14 days)
        min_valid_days: Minimum number of valid days required in sequence
        use_mask_channel: If True, add mask channel (1=valid, 0=missing)
        
    Returns:
        (X_sequences, y_targets, target_dates, sequence_metadata)
        - X_sequences: List of (seq_len, H, W, 173) input sequences (172 spectral + 1 mask)
        - y_targets: List of (H, W, 172) target maps
        - target_dates: List of target date strings
        - sequence_metadata: List of dicts with valid_count, valid_indices, etc.
    """
    logger.info("="*80)
    logger.info("CREATING SPARSE TEMPORAL SEQUENCES")
    logger.info("="*80)
    logger.info(f"Lookback window: {seq_len} days")
    logger.info(f"Min valid days required: {min_valid_days}")
    logger.info(f"Use mask channel: {use_mask_channel}")
    
    # Sort dates
    sorted_dates = sorted(daily_maps.keys())
    logger.info(f"Available dates: {len(sorted_dates)} ({sorted_dates[0]} to {sorted_dates[-1]})")
    
    X_sequences = []
    y_targets = []
    target_dates = []
    sequence_metadata = []
    
    # For each potential target date, create a lookback sequence
    for i in range(len(sorted_dates)):
        target_date_str = sorted_dates[i]
        target_date = datetime.strptime(target_date_str, '%Y%m%d')
        
        # Define the lookback window: [target - seq_len, ..., target - 1]
        lookback_dates = []
        for offset in range(seq_len, 0, -1):
            lookback_date = target_date - timedelta(days=offset)
            lookback_dates.append(lookback_date.strftime('%Y%m%d'))
        
        # Check which dates have valid data
        valid_mask = np.array([d in daily_maps for d in lookback_dates], dtype=np.float32)
        valid_count = int(valid_mask.sum())
        
        # Skip if insufficient valid days
        if valid_count < min_valid_days:
            continue
        
        # Build the sequence
        sequence = []
        for j, date_str in enumerate(lookback_dates):
            if valid_mask[j] == 1:
                # Valid day - use actual spectral map
                spectral_map = daily_maps[date_str]  # (H, W, 172)
                if use_mask_channel:
                    mask = np.ones((config.HEIGHT, config.WIDTH, 1), dtype=np.float32)
                    combined = np.concatenate([spectral_map, mask], axis=-1)  # (H, W, 173)
                else:
                    combined = spectral_map
            else:
                # Missing day - use zeros with mask=0
                spectral_map = np.zeros((config.HEIGHT, config.WIDTH, config.N_SPECTRAL_BANDS), dtype=np.float32)
                if use_mask_channel:
                    mask = np.zeros((config.HEIGHT, config.WIDTH, 1), dtype=np.float32)
                    combined = np.concatenate([spectral_map, mask], axis=-1)  # (H, W, 173)
                else:
                    combined = spectral_map
            
            sequence.append(combined)
        
        # Stack into array
        X_seq = np.stack(sequence, axis=0)  # (seq_len, H, W, 173 or 172)
        y_target = daily_maps[target_date_str]  # (H, W, 172)
        
        # Store metadata
        metadata = {
            'target_date': target_date_str,
            'lookback_dates': lookback_dates,
            'valid_mask': valid_mask,
            'valid_count': valid_count,
            'valid_indices': np.where(valid_mask == 1)[0].tolist(),
            'sparsity': 1.0 - (valid_count / seq_len),
            'most_recent_valid_day': None
        }
        
        # Find most recent valid day (for last-known-state fallback)
        for j in range(len(lookback_dates) - 1, -1, -1):
            if valid_mask[j] == 1:
                metadata['most_recent_valid_day'] = lookback_dates[j]
                metadata['days_since_last_valid'] = len(lookback_dates) - 1 - j
                break
        
        X_sequences.append(X_seq)
        y_targets.append(y_target)
        target_dates.append(target_date_str)
        sequence_metadata.append(metadata)
    
    logger.info(f"\nCreated {len(X_sequences)} sequences")
    
    if len(X_sequences) > 0:
        # Analyze sparsity distribution
        valid_counts = np.array([m['valid_count'] for m in sequence_metadata])
        sparsities = np.array([m['sparsity'] for m in sequence_metadata])
        
        logger.info(f"\nValid days per sequence:")
        logger.info(f"  Min: {valid_counts.min()}")
        logger.info(f"  Max: {valid_counts.max()}")
        logger.info(f"  Mean: {valid_counts.mean():.2f}")
        logger.info(f"  Median: {np.median(valid_counts):.0f}")
        
        logger.info(f"\nSparsity distribution:")
        logger.info(f"  Dense (≥12/14 days): {(valid_counts >= 12).sum()}")
        logger.info(f"  Moderate (8-11/14): {((valid_counts >= 8) & (valid_counts < 12)).sum()}")
        logger.info(f"  Sparse (4-7/14): {((valid_counts >= 4) & (valid_counts < 8)).sum()}")
        logger.info(f"  Very sparse (<4/14): {(valid_counts < 4).sum()}")
    
    return X_sequences, y_targets, target_dates, sequence_metadata


def compute_normalization_stats(sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute normalization statistics across all training sequences.
    
    Args:
        sequences: List of (seq_len, H, W, 172) arrays
        
    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (172,)
    """
    logger.info("Computing normalization statistics...")
    
    # Collect all valid spectral values
    all_values = {band: [] for band in range(config.N_SPECTRAL_BANDS)}
    
    for seq in tqdm(sequences, desc="Gathering stats"):
        for t in range(seq.shape[0]):
            for band in range(config.N_SPECTRAL_BANDS):
                values = seq[t, :, :, band]
                valid_values = values[~np.isnan(values)]
                all_values[band].extend(valid_values)
    
    # Compute mean and std for each band
    means = np.zeros(config.N_SPECTRAL_BANDS, dtype=np.float32)
    stds = np.zeros(config.N_SPECTRAL_BANDS, dtype=np.float32)
    
    for band in range(config.N_SPECTRAL_BANDS):
        if len(all_values[band]) > 0:
            means[band] = np.mean(all_values[band])
            stds[band] = np.std(all_values[band])
        else:
            means[band] = 0.0
            stds[band] = 1.0
    
    logger.info(f"Normalization stats computed:")
    logger.info(f"  Mean range: [{means.min():.6f}, {means.max():.6f}]")
    logger.info(f"  Std range: [{stds.min():.6f}, {stds.max():.6f}]")
    
    return {'mean': means, 'std': stds}


def save_sequences_to_hdf5(
    X: List[np.ndarray],
    y: List[np.ndarray],
    dates: List[str],
    metadata: List[Dict],
    output_path: Path,
    norm_stats: Optional[Dict[str, np.ndarray]] = None
):
    """
    Save sequences to HDF5 file for efficient loading.
    
    Args:
        X: List of (seq_len, H, W, 173) input sequences (172 spectral + 1 mask)
        y: List of (H, W, 172) targets
        dates: List of target date strings
        metadata: List of sequence metadata dictionaries
        output_path: Path to save HDF5 file
        norm_stats: Optional normalization statistics to include
    """
    logger.info(f"Saving {len(X)} sequences to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Save sequences
        f.create_dataset('X', data=np.array(X), compression='gzip', compression_opts=4)
        f.create_dataset('y', data=np.array(y), compression='gzip', compression_opts=4)
        
        # Save dates as strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('dates', data=dates, dtype=dt)
        
        # Save metadata
        meta_group = f.create_group('metadata')
        if len(metadata) > 0:
            meta_group.create_dataset('valid_counts', data=np.array([m['valid_count'] for m in metadata]))
            meta_group.create_dataset('sparsity', data=np.array([m['sparsity'] for m in metadata]))
            # Store lookback dates and valid masks as variable-length arrays
            dt_vlen = h5py.string_dtype(encoding='utf-8')
            meta_group.create_dataset('target_dates', data=[m['target_date'] for m in metadata], dtype=dt_vlen)
        
        # Save normalization stats if provided
        if norm_stats is not None:
            norm_group = f.create_group('normalization')
            norm_group.create_dataset('mean', data=norm_stats['mean'])
            norm_group.create_dataset('std', data=norm_stats['std'])
        
        # Save attributes
        f.attrs['n_sequences'] = len(X)
        f.attrs['seq_len'] = config.SEQ_LEN
        f.attrs['height'] = config.HEIGHT
        f.attrs['width'] = config.WIDTH
        f.attrs['n_spectral_bands'] = config.N_SPECTRAL_BANDS
        f.attrs['n_input_features'] = config.N_INPUT_FEATURES
        f.attrs['use_mask_channel'] = config.USE_MASK_CHANNEL
        f.attrs['min_valid_days'] = config.MIN_VALID_DAYS
        f.attrs['created'] = datetime.now().isoformat()
    
    logger.info(f"✅ Saved to {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    """
    Main pipeline: Process PACE granules → Match with GLERL → Create sequences → Save to HDF5.
    """
    logger.info("="*80)
    logger.info("PHASE 7 DATA PREPARATION")
    logger.info("="*80)
    
    # Step 1: Create spatial grid
    grid_lats, grid_lons = create_spatial_grid()
    logger.info(f"Spatial grid: {len(grid_lats)}x{len(grid_lons)} ({config.HEIGHT}x{config.WIDTH})")
    
    # Step 2: Load GLERL ground truth measurements
    glerl_df = load_glerl_measurements()
    
    # Step 3: Process all PACE granules to daily maps
    daily_maps = process_all_pace_granules()
    
    # Step 4: Match GLERL measurements to PACE spectra
    matched_samples = match_glerl_to_pace_spectra(glerl_df, daily_maps, grid_lats, grid_lons)
    
    # Save matched samples for supervised training
    matched_df = pd.DataFrame(matched_samples)
    matched_csv_path = config.SPECTRAL_DATA_DIR / 'glerl_pace_matched.csv'
    matched_df.to_csv(matched_csv_path, index=False)
    logger.info(f"\n✅ Saved {len(matched_df)} matched samples to {matched_csv_path}")
    
    # Also save as numpy for quick loading during training
    matched_npz_path = config.SPECTRAL_DATA_DIR / 'glerl_pace_matched.npz'
    np.savez(
        matched_npz_path,
        spectra=np.array([s['spectra'] for s in matched_samples]),
        mc_binary=np.array([s['mc_binary'] for s in matched_samples]),
        mc_value=np.array([s['mc_value'] for s in matched_samples]),
        glerl_dates=np.array([s['glerl_date_str'] for s in matched_samples]),
        pace_dates=np.array([s['pace_date_str'] for s in matched_samples]),
        temporal_offsets=np.array([s['temporal_offset_days'] for s in matched_samples]),
        lats=np.array([s['lat'] for s in matched_samples]),
        lons=np.array([s['lon'] for s in matched_samples]),
        grid_rows=np.array([s['grid_row'] for s in matched_samples]),
        grid_cols=np.array([s['grid_col'] for s in matched_samples]),
        spatial_distances=np.array([s['spatial_distance_km'] for s in matched_samples])
    )
    logger.info(f"✅ Saved matched samples to {matched_npz_path}")
    
    # Step 5: Create patch-based sequences for supervised training
    logger.info(f"\n{'='*80}")
    logger.info("CREATING PATCH-BASED SEQUENCES FOR SUPERVISED TRAINING")
    logger.info(f"{'='*80}")
    
    X_patch, y_patch, metadata_patch = create_patch_based_sequences(
        matched_samples, daily_maps
    )
    
    # Step 6: Stratified temporal split for patch sequences
    # Sort by timestamp to ensure temporal ordering, then use stratified sampling
    # Every 7th sample → validation, every 10th → test, rest → training
    train_X_patch, train_y_patch, train_meta_patch = [], [], []
    val_X_patch, val_y_patch, val_meta_patch = [], [], []
    test_X_patch, test_y_patch, test_meta_patch = [], [], []
    
    # Sort patches by PACE date to maintain temporal order
    sorted_indices = sorted(
        range(len(X_patch)),
        key=lambda i: datetime.strptime(metadata_patch[i]['pace_date'], '%Y%m%d')
    )
    
    # Stratified sampling: every 7th→val, every 10th→test
    for idx_position, idx in enumerate(sorted_indices):
        if idx_position % 10 == 6:  # Every 7th (0-indexed: 6, 16, 26, ...)
            val_X_patch.append(X_patch[idx])
            val_y_patch.append(y_patch[idx])
            val_meta_patch.append(metadata_patch[idx])
        elif idx_position % 10 == 9:  # Every 10th (0-indexed: 9, 19, 29, ...)
            test_X_patch.append(X_patch[idx])
            test_y_patch.append(y_patch[idx])
            test_meta_patch.append(metadata_patch[idx])
        else:  # Remaining 80% → training
            train_X_patch.append(X_patch[idx])
            train_y_patch.append(y_patch[idx])
            train_meta_patch.append(metadata_patch[idx])
    
    logger.info(f"\nStratified temporal split (patch sequences):")
    logger.info(f"  Train: {len(train_X_patch)} patch sequences ({100*len(train_X_patch)/len(X_patch):.1f}%)")
    logger.info(f"  Val:   {len(val_X_patch)} patch sequences ({100*len(val_X_patch)/len(X_patch):.1f}%)")
    logger.info(f"  Test:  {len(test_X_patch)} patch sequences ({100*len(test_X_patch)/len(X_patch):.1f}%)")
    
    # Class distribution per split
    for split_name, y_split in [('Train', train_y_patch), ('Val', val_y_patch), ('Test', test_y_patch)]:
        if len(y_split) > 0:
            y_arr = np.array(y_split)
            logger.info(f"  {split_name} class balance: Neg={( y_arr==0).sum()}, Pos={(y_arr==1).sum()}")
    
    # Step 7: Compute normalization stats from training patches
    # Extract spectral values from all training patches
    logger.info("\nComputing normalization statistics from training patches...")
    all_train_spectral = []
    for patch_seq in train_X_patch:
        # patch_seq is (seq_len, patch_size, patch_size, 173)
        # Extract spectral bands (first 172 channels)
        spectral_data = patch_seq[:, :, :, :config.N_SPECTRAL_BANDS]
        all_train_spectral.append(spectral_data)
    
    if len(all_train_spectral) > 0:
        all_train_spectral = np.concatenate(all_train_spectral, axis=0)  # (N*seq_len, patch, patch, 172)
        
        # Compute per-band statistics
        norm_means = np.zeros(config.N_SPECTRAL_BANDS, dtype=np.float32)
        norm_stds = np.zeros(config.N_SPECTRAL_BANDS, dtype=np.float32)
        
        for band in range(config.N_SPECTRAL_BANDS):
            band_values = all_train_spectral[:, :, :, band].ravel()
            # Filter out zeros (masked pixels)
            band_values = band_values[band_values != 0]
            if len(band_values) > 0:
                norm_means[band] = np.mean(band_values)
                norm_stds[band] = np.std(band_values)
            else:
                norm_means[band] = 0.0
                norm_stds[band] = 1.0
        
        logger.info(f"✅ Computed normalization stats from {len(all_train_spectral)} training frames")
    else:
        logger.warning("No training patches available - using default normalization")
        norm_means = np.zeros(config.N_SPECTRAL_BANDS, dtype=np.float32)
        norm_stds = np.ones(config.N_SPECTRAL_BANDS, dtype=np.float32)
    
    # Step 8: Save patch sequences to NPZ files
    patch_train_path = config.SPECTRAL_DATA_DIR / 'glerl_patch_sequences_train.npz'
    np.savez(
        patch_train_path,
        X=np.array(train_X_patch) if len(train_X_patch) > 0 else np.array([]),
        y=np.array(train_y_patch) if len(train_y_patch) > 0 else np.array([]),
        metadata=train_meta_patch
    )
    logger.info(f"✅ Saved {len(train_X_patch)} training patch sequences to {patch_train_path}")
    
    patch_val_path = config.SPECTRAL_DATA_DIR / 'glerl_patch_sequences_val.npz'
    np.savez(
        patch_val_path,
        X=np.array(val_X_patch) if len(val_X_patch) > 0 else np.array([]),
        y=np.array(val_y_patch) if len(val_y_patch) > 0 else np.array([]),
        metadata=val_meta_patch
    )
    logger.info(f"✅ Saved {len(val_X_patch)} validation patch sequences to {patch_val_path}")
    
    patch_test_path = config.SPECTRAL_DATA_DIR / 'glerl_patch_sequences_test.npz'
    np.savez(
        patch_test_path,
        X=np.array(test_X_patch) if len(test_X_patch) > 0 else np.array([]),
        y=np.array(test_y_patch) if len(test_y_patch) > 0 else np.array([]),
        metadata=test_meta_patch
    )
    logger.info(f"✅ Saved {len(test_X_patch)} test patch sequences to {patch_test_path}")
    
    # Save normalization stats
    norm_stats_path = config.get_normalization_stats_path('train')
    np.savez(
        norm_stats_path,
        mean=norm_means,
        std=norm_stds
    )
    logger.info(f"✅ Saved normalization stats to {norm_stats_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("DATA PREPARATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {config.SPECTRAL_DATA_DIR}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  1. glerl_pace_matched.csv/npz - {len(matched_samples)} point-level GLERL-PACE matches")
    logger.info(f"  2. glerl_patch_sequences_train.npz - {len(train_X_patch)} patch sequences ({config.PATCH_SIZE}x{config.PATCH_SIZE})")
    logger.info(f"  3. glerl_patch_sequences_val.npz - {len(val_X_patch)} validation patch sequences")
    logger.info(f"  4. glerl_patch_sequences_test.npz - {len(test_X_patch)} test patch sequences")
    logger.info(f"  5. normalization_stats_train.npz - Per-band spectral normalization")
    logger.info(f"\nReady for Phase 7 Supervised Training:")
    logger.info(f"  Input shape: ({config.SEQ_LEN}, {config.PATCH_SIZE}, {config.PATCH_SIZE}, {config.N_INPUT_FEATURES})")
    logger.info(f"  Output: Binary classification (center pixel MC ≥ {config.MC_THRESHOLD} µg/L)")
    logger.info(f"  Training samples: {len(train_X_patch)}")
    logger.info(f"  Spatial context: ~{config.PATCH_SIZE * config.RES_KM:.1f}km x {config.PATCH_SIZE * config.RES_KM:.1f}km per patch")


if __name__ == '__main__':
    main()
