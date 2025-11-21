"""
Data collection module for microcystin detection training data.

This module handles:
- Downloading PACE/Sentinel-3 granules from NASA Earthdata
- Processing granules to extract patches around ground truth locations
- Implementing temporal splitting to prevent data leakage
- Saving features and labels for model training
- Hash-based tracking to prevent duplicate samples and enable crash recovery
"""

import os
import logging
import hashlib
from typing import List, Tuple, Set, Optional, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import earthaccess

from . import config
from .utils import (
    with_retries,
    extract_datetime_from_filename,
    estimate_position,
    process_pace_granule,
    extract_pace_patch,
    configure_logging,
    load_station_colors,
    save_station_colors,
    get_granule_filename
)


def compute_sample_hash(filename: str, station: str, timestamp: datetime, patch_size: int) -> str:
    """
    Compute unique hash for a sample to prevent duplicates.
    
    Args:
        filename: Granule filename
        station: Station name
        timestamp: Observation timestamp
        patch_size: Patch size used
        
    Returns:
        SHA256 hash string (first 16 chars)
    """
    # Create unique string from sample identifying info
    unique_str = f"{filename}|{station}|{timestamp.isoformat()}|{patch_size}"
    hash_obj = hashlib.sha256(unique_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def build_existing_hashes(results: List) -> Set[str]:
    """
    Build set of hashes from existing results.
    
    Args:
        results: List of existing samples (filename, station, label_tuple, ...)
        
    Returns:
        Set of sample hashes
    """
    hashes = set()
    for sample in results:
        if len(sample) >= 6:
            filename, station, label_tuple, _, _, patch_size = sample[:6]
            # Extract timestamp from label_tuple: (station, timestamp, lat, lon, ...)
            if len(label_tuple) >= 2:
                timestamp = label_tuple[1]
                sample_hash = compute_sample_hash(filename, station, timestamp, patch_size)
                hashes.add(sample_hash)
    return hashes


class CorruptedGranulesTracker:
    """Track and persist list of corrupted granule filenames."""
    
    def __init__(self, filepath: str):
        """
        Initialize tracker.
        
        Args:
            filepath: Path to corrupted_granules.txt
        """
        self.filepath = filepath
        self.corrupted: Set[str] = self._load()
    
    def _load(self) -> Set[str]:
        """Load corrupted granules from file."""
        if not os.path.exists(self.filepath):
            return set()
        
        with open(self.filepath, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def mark_corrupted(self, filename: str) -> None:
        """
        Mark a granule as corrupted and persist to file.
        
        Args:
            filename: Granule filename
        """
        if filename not in self.corrupted:
            self.corrupted.add(filename)
            with open(self.filepath, 'a') as f:
                f.write(filename + '\n')
            logging.info(f"Marked granule as corrupted: {filename}")
    
    def is_corrupted(self, filename: str) -> bool:
        """Check if granule is corrupted."""
        return filename in self.corrupted


def load_ground_truth_data(
    glerl_csv: str,
    user_labels_csv: Optional[str] = None,
    pm_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Load GLERL ground truth data with optional user labels.
    
    Args:
        glerl_csv: Path to glrl-hab-data.csv
        user_labels_csv: Path to user-labels.csv (optional)
        pm_threshold: Threshold for assigning positive class to user labels
        
    Returns:
        DataFrame with columns: timestamp, lat, lon, station_name, 
                                particulate_microcystin, etc.
    """
    # Load GLERL data
    df = pd.read_csv(glerl_csv, index_col=0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    logging.info(f"Loaded {len(df)} GLERL observations from {glerl_csv}")
    
    # Optionally append user labels
    if user_labels_csv and os.path.exists(user_labels_csv):
        user_df = pd.read_csv(user_labels_csv)
        user_df['timestamp'] = pd.to_datetime(user_df['date'], utc=True)
        user_df['station_name'] = [f"USER_{i}" for i in range(len(user_df))]
        
        # Convert binary labels to microcystin concentration
        user_df.loc[user_df['label'] == 'negative', 'particulate_microcystin'] = 0.0
        user_df.loc[user_df['label'] == 'positive', 'particulate_microcystin'] = pm_threshold
        
        df = pd.concat([df, user_df], ignore_index=True)
        logging.info(f"Added {len(user_df)} user-labeled samples")
    
    return df


def get_temporal_split(
    df: pd.DataFrame,
    split_name: str = 'train'
) -> pd.DataFrame:
    """
    Filter DataFrame to specific temporal split (train/val/test).
    
    Uses config.TEMPORAL_SPLIT to determine which dates belong to each split.
    
    Args:
        df: DataFrame with 'timestamp' column
        split_name: One of 'train', 'val', 'test'
        
    Returns:
        Filtered DataFrame
    """
    if split_name not in config.TEMPORAL_SPLIT:
        raise ValueError(f"Unknown split '{split_name}'. Use 'train', 'val', or 'test'.")
    
    split_dates = config.TEMPORAL_SPLIT[split_name]
    
    # Convert to datetime if needed
    split_datetimes = [
        pd.to_datetime(d, utc=True) if isinstance(d, str) else d
        for d in split_dates
    ]
    
    # Filter to dates within temporal split
    # Use date-only comparison to match any observation on these dates
    df_filtered = df[df['timestamp'].dt.date.isin(
        [dt.date() for dt in split_datetimes]
    )]
    
    logging.info(f"Temporal split '{split_name}': {len(df)} → {len(df_filtered)} samples "
                 f"across {len(split_dates)} dates")
    
    return df_filtered


def process_single_granule(
    filepath: str,
    station_df: pd.DataFrame,
    bbox: Tuple[float, float, float, float],
    res_km: float,
    wavelengths: np.ndarray,
    patch_sizes: List[int],
    half_time_window: int,
    corrupted_tracker: CorruptedGranulesTracker,
    existing_hashes: Set[str] = None
) -> List[Tuple]:
    """
    Process one granule and extract patches for all patch sizes.
    
    Args:
        filepath: Path to downloaded granule file
        station_df: DataFrame of ground truth observations
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        res_km: Grid resolution in km
        wavelengths: Array of wavelengths to process
        patch_sizes: List of patch sizes to extract
        half_time_window: Half-width of time window in days
        corrupted_tracker: Tracker for corrupted granules
        existing_hashes: Set of already-processed sample hashes (for deduplication)
        
    Returns:
        List of sample tuples
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        res_km: Resolution in kilometers
        wavelengths: Array of wavelengths to process
        patch_sizes: List of patch sizes to extract (e.g., [3, 5, 7, 9])
        half_time_window: Half-width of time window in days
        corrupted_tracker: Tracker for corrupted granules
        
    Returns:
        List of tuples: (filename, station, label_tuple, feature_vector, 
                        valid_fraction, patch_size)
    """
    filename = os.path.basename(filepath)
    
    # Skip if corrupted
    if corrupted_tracker.is_corrupted(filename):
        logging.debug(f"Skipping corrupted granule: {filename}")
        return []
    
    # Extract datetime
    dt = extract_datetime_from_filename(filename)
    if dt is None:
        logging.warning(f"Cannot parse datetime from {filename}")
        return []
    
    # Process granule
    try:
        result = process_pace_granule(filepath, bbox, wavelengths, res_km)
        if result is None:
            logging.warning(f"Failed to process {filename}")
            return []
        
        wls, arr_stack, target_lats, target_lons = result
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        corrupted_tracker.mark_corrupted(filename)
        return []
    
    # Compute global mean reflectance per wavelength (context feature)
    global_means = np.nanmean(arr_stack, axis=(1, 2))
    
    # Find observations within time window
    t0 = pd.to_datetime(dt).tz_localize('UTC')
    window_start = t0 - timedelta(days=half_time_window)
    window_end = t0 + timedelta(days=half_time_window)
    
    df_window = station_df[
        (station_df.timestamp >= window_start) &
        (station_df.timestamp <= window_end)
    ]
    
    if df_window.empty:
        logging.debug(f"No observations within {half_time_window} days of {filename}")
        return []
    
    # Extract patches for each station
    results = []
    
    for station in df_window.station_name.unique():
        df_station = df_window[df_window.station_name == station]
        
        # Estimate station position at granule time
        lat0, lon0 = estimate_position(
            df_station.timestamp.tolist(),
            df_station.lat.values,
            df_station.lon.values,
            t0
        )
        
        # Extract patches for all patch sizes
        for patch_size in patch_sizes:
            # Check if this sample already exists (hash-based deduplication)
            if existing_hashes is not None:
                sample_hash = compute_sample_hash(filename, station, t0, patch_size)
                if sample_hash in existing_hashes:
                    logging.debug(f"Skipping duplicate: {filename}, {station}, {t0}, patch={patch_size}")
                    continue
            
            # Extract patch
            patch_dict = extract_pace_patch(
                arr_stack, wls, lon0, lat0,
                pixel_count=patch_size,
                lat_centers=target_lats,
                lon_centers=target_lons
            )
            
            if patch_dict is None:
                continue
            
            # Compute coverage fraction
            total_pixels = sum(arr.size for arr in patch_dict.values())
            valid_pixels = sum(
                np.count_nonzero(~np.isnan(arr)) for arr in patch_dict.values()
            )
            valid_frac = valid_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Skip patches with insufficient valid pixel coverage
            if valid_frac < config.MIN_VALID_FRACTION:
                logging.debug(f"Skipping patch with low coverage: {valid_frac:.2f} < {config.MIN_VALID_FRACTION} "
                             f"(station={station}, patch_size={patch_size})")
                continue
            
            # Build feature vector: [patch_pixels (flattened), global_means]
            patch_stack = np.stack(
                [patch_dict[wl] for wl in sorted(patch_dict)],
                axis=-1
            )  # Shape: (patch_size, patch_size, n_wavelengths)
            
            feature_vector = np.concatenate([
                patch_stack.flatten(),
                global_means
            ])
            
            # Build label tuple
            label_tuple = (
                station,
                t0,
                lat0,
                lon0,
                df_station.particulate_microcystin.mean(),
                df_station.dissolved_microcystin.mean() 
                    if 'dissolved_microcystin' in df_station else np.nan,
                df_station.extracted_chla.mean() 
                    if 'extracted_chla' in df_station else np.nan
            )
            
            # Record: (filename, station, label_tuple, feature_vector, valid_frac, patch_size)
            results.append((
                filename,
                station,
                label_tuple,
                feature_vector,
                valid_frac,
                patch_size
            ))
    
    return results


def collect_training_data(
    sensor: str = 'PACE',
    patch_sizes: List[int] = None,
    half_time_window: int = 2,
    pm_threshold: float = 0.1,
    data_dir: str = './',
    temporal_split: str = 'train',
    load_user_labels: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> None:
    """
    Main data collection pipeline with temporal splitting.
    
    This function:
    1. Loads ground truth data (GLERL + optional user labels)
    2. Applies temporal split (train/val/test)
    3. Downloads relevant satellite granules
    4. Extracts patches for all patch sizes
    5. Saves features and labels
    
    Args:
        sensor: Sensor name ('PACE' or 'Sentinel-3')
        patch_sizes: List of patch sizes (default from config)
        half_time_window: Half-width of time window in days
        pm_threshold: Threshold for positive class (µg/L)
        data_dir: Directory containing data files and for saving output
        temporal_split: Which split to collect ('train', 'val', 'test', or 'all')
        load_user_labels: Whether to include user-labeled data
        start_date: Override start date (default from config)
        end_date: Override end date (default from config)
    """
    configure_logging()
    
    if patch_sizes is None:
        patch_sizes = config.PATCH_SIZES
    
    sensor = sensor.upper()
    sensor_params = config.SENSOR_PARAMS[sensor]
    
    logging.info(f"Starting data collection: sensor={sensor}, split={temporal_split}, "
                 f"patch_sizes={patch_sizes}")
    
    # ===== LOAD GROUND TRUTH =====
    glerl_csv = os.path.join(data_dir, 'glrl-hab-data.csv')
    user_csv = os.path.join(data_dir, 'user-labels.csv') if load_user_labels else None
    
    df_all = load_ground_truth_data(glerl_csv, user_csv, pm_threshold)
    
    # ===== APPLY TEMPORAL SPLIT =====
    if temporal_split == 'all':
        df = df_all
        logging.info("Collecting data for all splits")
    else:
        df = get_temporal_split(df_all, temporal_split)
    
    if df.empty:
        logging.warning("No observations in temporal split; exiting")
        return
    
    # ===== DETERMINE DATE RANGE =====
    if end_date is None:
        end_date = df['timestamp'].max()
        if pd.isna(end_date):
            logging.error("No valid timestamps in data")
            return
        end_date = end_date.date()
    
    if start_date is None:
        # Use sensor-specific start date or earliest observation
        sensor_start = sensor_params.get('start_date', '2024-04-16')
        if isinstance(sensor_start, str):
            start_date = datetime.strptime(sensor_start, '%Y-%m-%d').date()
        elif isinstance(sensor_start, datetime):
            start_date = sensor_start.date()
        else:
            start_date = sensor_start
    
    if start_date > end_date:
        logging.info("No data window to process")
        return
    
    logging.info(f"Date range: {start_date} to {end_date}")
    
    # ===== SETUP PATHS =====
    extraction_path = os.path.join(
        data_dir,
        f'training_data_{sensor}_{temporal_split}.npy'
    )
    corrupted_path = os.path.join(data_dir, 'corrupted_granules.txt')
    station_colors_json = os.path.join(data_dir, f'station_colors_{sensor}.json')
    
    # Initialize tracker and results
    corrupted_tracker = CorruptedGranulesTracker(corrupted_path)
    
    if os.path.exists(extraction_path):
        existing = np.load(extraction_path, allow_pickle=True)
        results = existing.tolist()
        logging.info(f"Loaded {len(results)} existing samples from {extraction_path}")
        # Build hash set from existing samples for deduplication
        existing_hashes = build_existing_hashes(results)
        logging.info(f"Built hash set with {len(existing_hashes)} existing samples")
    else:
        results = []
        existing_hashes = set()
    
    station_colors = load_station_colors(station_colors_json)
    
    # ===== AUTHENTICATE WITH EARTHDATA =====
    auth = earthaccess.login(persist=True)
    
    # ===== GET WAVELENGTHS FROM REFERENCE FILE =====
    logging.info("Retrieving wavelength list from reference file...")
    ref_search = with_retries(
        earthaccess.search_data,
        short_name="PACE_OCI_L2_AOP",
        temporal=("2024-06-01", "2024-06-05"),
        bounding_box=sensor_params['bbox']
    )
    
    if not ref_search:
        logging.error("Could not find reference PACE file for wavelengths")
        return
    
    ref_file = with_retries(earthaccess.download, ref_search, "./data")[0]
    wavelengths = xr.open_dataset(
        ref_file,
        group="sensor_band_parameters"
    )["wavelength_3d"].data
    
    logging.info(f"Loaded {len(wavelengths)} wavelengths")
    
    # ===== PROCESS DATA BY MONTH =====
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    data_root = os.path.join('./data', sensor)
    os.makedirs(data_root, exist_ok=True)
    
    for month_start in month_starts.to_pydatetime():
        year, month = month_start.year, month_start.month
        
        # Determine month end
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        if next_month.date() > end_date:
            next_month = datetime(
                end_date.year,
                end_date.month,
                end_date.day
            ) + timedelta(days=1)
        
        start_iso = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = next_month.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logging.info(f"Searching {sensor} granules for {year}-{month:02d}")
        
        # Search for granules
        all_granules = []
        for short_name in sensor_params['short_names']:
            try:
                granules = with_retries(
                    earthaccess.search_data,
                    short_name=short_name,
                    temporal=(start_iso, end_iso),
                    bounding_box=sensor_params['bbox']
                )
                
                if granules:
                    logging.info(f"Found {len(granules)} granules for {short_name}")
                    all_granules.extend(granules)
            except Exception as e:
                logging.error(f"Search failed for {short_name}: {e}")
        
        if not all_granules:
            logging.info(f"No granules found for {year}-{month:02d}")
            continue
        
        # Filter granules to those with nearby observations
        filtered_granules = []
        for item in all_granules:
            filename = get_granule_filename(item)
            if filename is None:
                continue
            
            # Skip corrupted
            if corrupted_tracker.is_corrupted(filename):
                continue
            
            # Get granule datetime
            dt = None
            if hasattr(item, 'time_start'):
                try:
                    dt = pd.to_datetime(item.time_start, utc=True)
                except:
                    dt = None
            
            if dt is None:
                dt_parsed = extract_datetime_from_filename(filename)
                if dt_parsed:
                    dt = pd.to_datetime(dt_parsed).tz_localize('UTC')
            
            if dt is None:
                logging.debug(f"Cannot determine datetime for {filename}")
                continue
            
            # Check for nearby observations
            window_start = dt - timedelta(days=half_time_window)
            window_end = dt + timedelta(days=half_time_window)
            
            has_obs = (
                (df['timestamp'] >= window_start) &
                (df['timestamp'] <= window_end)
            ).any()
            
            if has_obs:
                filtered_granules.append(item)
        
        logging.info(f"Filtered to {len(filtered_granules)} granules with nearby observations")
        
        if not filtered_granules:
            continue
        
        # Download granules
        try:
            logging.info("Downloading granules...")
            paths = with_retries(
                earthaccess.download,
                filtered_granules,
                './data'
            )
        except Exception as e:
            logging.error(f"Download failed: {e}")
            continue
        
        # Process each granule
        for filepath in paths:
            granule_results = process_single_granule(
                filepath=filepath,
                station_df=df,
                bbox=sensor_params['bbox'],
                res_km=sensor_params['res_km'],
                wavelengths=wavelengths,
                patch_sizes=patch_sizes,
                half_time_window=half_time_window,
                corrupted_tracker=corrupted_tracker,
                existing_hashes=existing_hashes
            )
            
            # Add new hashes to set (for deduplication within same run)
            for sample in granule_results:
                if len(sample) >= 6:
                    filename, station, label_tuple, _, _, patch_size = sample[:6]
                    if len(label_tuple) >= 2:
                        timestamp = label_tuple[1]
                        sample_hash = compute_sample_hash(filename, station, timestamp, patch_size)
                        existing_hashes.add(sample_hash)
            
            results.extend(granule_results)
            
            # Save incrementally
            if len(granule_results) > 0:
                np.save(extraction_path, np.array(results, dtype=object))
                logging.info(f"Saved {len(results)} total samples to {extraction_path}")
    
    # ===== FINAL SAVE =====
    np.save(extraction_path, np.array(results, dtype=object))
    save_station_colors(station_colors_json, station_colors)
    
    logging.info(f"Data collection complete: {len(results)} patches saved to {extraction_path}")
    
    # Log statistics
    if results:
        patch_size_counts = {}
        for _, _, _, _, _, ps in results:
            patch_size_counts[ps] = patch_size_counts.get(ps, 0) + 1
        
        logging.info("Samples per patch size:")
        for ps, count in sorted(patch_size_counts.items()):
            logging.info(f"  {ps}×{ps}: {count} samples")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data for microcystin detection')
    parser.add_argument('--sensor', type=str, default='PACE',
                        choices=['PACE', 'Sentinel-3'],
                        help='Sensor type')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test', 'all'],
                        help='Temporal split to collect')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing ground truth data (default: from config)')
    parser.add_argument('--patch-sizes', type=int, nargs='+',
                        default=None,
                        help='Patch sizes to extract (default: from config)')
    parser.add_argument('--time-window', type=int, default=2,
                        help='Half-width of time window in days')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='PM threshold for binary classification')
    parser.add_argument('--user-labels', action='store_true',
                        help='Include user-labeled data')
    
    args = parser.parse_args()
    
    # Use config directory if not specified
    data_dir = args.data_dir if args.data_dir is not None else str(config.BASE_DIR)
    
    collect_training_data(
        sensor=args.sensor,
        patch_sizes=args.patch_sizes,
        half_time_window=args.time_window,
        pm_threshold=args.threshold,
        data_dir=data_dir,
        temporal_split=args.split,
        load_user_labels=args.user_labels
    )
