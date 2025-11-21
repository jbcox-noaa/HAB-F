"""
Data balancing module for microcystin detection.

This module handles class imbalance by collecting additional negative samples
from winter months (when microcystin is typically absent).
"""

import os
import logging
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import earthaccess

from . import config
from .utils import (
    with_retries,
    extract_datetime_from_filename,
    process_pace_granule,
    extract_pace_patch,
    get_granule_filename,
    configure_logging
)


def analyze_class_distribution(
    training_data: np.ndarray,
    pm_threshold: float = 0.1
) -> Dict[str, int]:
    """
    Analyze class distribution in training data.
    
    Args:
        training_data: Array of training samples
        pm_threshold: Threshold for positive class
        
    Returns:
        Dictionary with 'positive' and 'negative' counts
    """
    positive = 0
    negative = 0
    
    for sample in training_data:
        # sample format: (filename, station, label_tuple, features, valid_frac, patch_size)
        label_tuple = sample[2]
        pm_concentration = label_tuple[4]  # particulate_microcystin
        
        if pm_concentration >= pm_threshold:
            positive += 1
        else:
            negative += 1
    
    return {
        'positive': positive,
        'negative': negative,
        'total': len(training_data),
        'imbalance_ratio': positive / negative if negative > 0 else float('inf')
    }


def get_winter_months(start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Get list of winter month ranges between start and end dates.
    
    Winter months: December, January, February, March
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        
    Returns:
        List of (month_start, month_end) tuples
    """
    winter_ranges = []
    
    # Generate all months in range
    current = start_date.replace(day=1)
    while current <= end_date:
        # Check if winter month (Dec=12, Jan=1, Feb=2, Mar=3)
        if current.month in [12, 1, 2, 3]:
            # Determine next month
            if current.month == 12:
                next_month = datetime(current.year + 1, 1, 1)
            else:
                next_month_num = current.month + 1
                if next_month_num > 12:
                    next_month = datetime(current.year + 1, 1, 1)
                else:
                    next_month = datetime(current.year, next_month_num, 1)
            
            # Don't exceed end_date
            if current <= end_date:
                winter_ranges.append((current, min(next_month, end_date)))
        
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return winter_ranges


def balance_by_oversampling_negatives(
    sensor: str = 'PACE',
    patch_size: int = 3,
    pm_threshold: float = 0.1,
    data_dir: str = './',
    n_negative_samples: int = 500,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    random_seed: int = 42
) -> None:
    """
    Balance training data by adding negative samples from winter months.
    
    Strategy:
    1. Load existing training data
    2. Analyze class distribution
    3. If imbalanced (pos > neg), download winter granules
    4. Extract patches at random locations in Lake Erie
    5. Add as negative samples (PM = 0.0)
    6. Stop when classes are balanced
    
    Args:
        sensor: Sensor name ('PACE' or 'Sentinel-3')
        patch_size: Patch size in pixels
        pm_threshold: Threshold for positive class
        data_dir: Directory containing training data
        n_negative_samples: Target number of negative samples to add
        start_date: Start date for winter granule search
        end_date: End date for winter granule search
        random_seed: Random seed for reproducibility
    """
    configure_logging()
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    sensor = sensor.upper()
    sensor_params = config.SENSOR_PARAMS[sensor]
    
    logging.info(f"Starting class balancing: sensor={sensor}, patch_size={patch_size}")
    
    # ===== LOAD EXISTING DATA =====
    training_path = os.path.join(data_dir, f'training_data_{sensor}.npy')
    
    if not os.path.exists(training_path):
        logging.error(f"Training data not found at {training_path}")
        logging.error("Run data_collection.py first")
        return
    
    training_data = np.load(training_path, allow_pickle=True)
    results = training_data.tolist()
    
    logging.info(f"Loaded {len(results)} existing samples")
    
    # ===== ANALYZE DISTRIBUTION =====
    dist = analyze_class_distribution(results, pm_threshold)
    logging.info(f"Current distribution: {dist['positive']} positive, {dist['negative']} negative")
    logging.info(f"Imbalance ratio: {dist['imbalance_ratio']:.2f}")
    
    # Target: 2x as many negatives as positives (ratio = 0.5)
    target_ratio = 0.5
    target_negatives = int(dist['positive'] / target_ratio)
    
    if dist['negative'] >= target_negatives:
        logging.info(f"Classes already balanced (have {dist['negative']}, need {target_negatives}); no action needed")
        return
    
    n_needed = target_negatives - dist['negative']
    n_to_collect = min(n_negative_samples, n_needed) if n_negative_samples < n_needed else n_needed
    
    logging.info(f"Target: {target_negatives} negatives (2x positives) for ratio=0.5")
    logging.info(f"Need {n_needed} more negative samples; will attempt to collect up to {n_to_collect}")
    
    # ===== SET DATE RANGE =====
    if start_date is None:
        start_date = datetime.strptime(config.OVERSAMPLE_START_DATE, '%Y-%m-%d')
    if end_date is None:
        end_date = datetime.strptime(config.OVERSAMPLE_END_DATE, '%Y-%m-%d')
    
    logging.info(f"Winter date range: {start_date.date()} to {end_date.date()}")
    
    # ===== AUTHENTICATE =====
    auth = earthaccess.login(persist=True)
    
    # ===== GET WAVELENGTHS =====
    logging.info("Retrieving wavelength list...")
    ref_search = with_retries(
        earthaccess.search_data,
        short_name="PACE_OCI_L2_AOP",
        temporal=("2024-06-01", "2024-06-05"),
        bounding_box=sensor_params['bbox']
    )
    
    if not ref_search:
        logging.error("Could not find reference file for wavelengths")
        return
    
    ref_file = with_retries(earthaccess.download, ref_search, "./data")[0]
    wavelengths = xr.open_dataset(
        ref_file,
        group="sensor_band_parameters"
    )["wavelength_3d"].data
    
    # ===== SEARCH WINTER GRANULES =====
    winter_ranges = get_winter_months(start_date, end_date)
    all_granules = []
    
    for month_start, month_end in winter_ranges:
        start_iso = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = month_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logging.info(f"Searching winter granules: {month_start.strftime('%Y-%m')}")
        
        for short_name in sensor_params['short_names']:
            try:
                granules = with_retries(
                    earthaccess.search_data,
                    short_name=short_name,
                    temporal=(start_iso, end_iso),
                    bounding_box=sensor_params['bbox']
                )
                
                if granules:
                    all_granules.extend(granules)
                    logging.info(f"Found {len(granules)} granules for {short_name}")
            except Exception as e:
                logging.error(f"Search failed for {short_name}: {e}")
    
    if not all_granules:
        logging.warning("No winter granules found")
        return
    
    # Deduplicate
    unique_granules = []
    seen_filenames = set()
    for item in all_granules:
        filename = get_granule_filename(item)
        if filename and filename not in seen_filenames:
            unique_granules.append(item)
            seen_filenames.add(filename)
    
    logging.info(f"Found {len(unique_granules)} unique winter granules")
    
    # Shuffle
    random.shuffle(unique_granules)
    
    # ===== PROCESS GRANULES =====
    n_collected = 0
    temp_dir = os.path.join('./data', sensor, 'balance_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define random sampling locations within Lake Erie bbox
    bbox = sensor_params['bbox']
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Track sampled locations per granule to avoid duplicates across iterations
    sampled_locations = {}  # {filename: set of (lat, lon) tuples}
    
    # We may need to loop through all granules multiple times
    max_iterations = 5
    iteration = 0
    
    while n_collected < n_to_collect and iteration < max_iterations:
        iteration += 1
        logging.info(f"Starting iteration {iteration}/{max_iterations}")
        
        # Shuffle granules at start of each iteration for different processing order
        random.shuffle(unique_granules)
        
        for granule in unique_granules:
            if n_collected >= n_to_collect:
                break
            
            filename = get_granule_filename(granule)
            if not filename:
                continue
            
            # Initialize sampled locations for this granule if needed
            if filename not in sampled_locations:
                sampled_locations[filename] = set()
            
            logging.info(f"Processing {filename} (iteration {iteration})...")
            
            # Download
            try:
                paths = with_retries(earthaccess.download, [granule], temp_dir)
                if not paths:
                    continue
                granule_path = paths[0]
            except Exception as e:
                logging.warning(f"Download failed: {e}")
                continue
            
            # Process granule
            try:
                result = process_pace_granule(
                    granule_path,
                    bbox,
                    wavelengths,
                    sensor_params['res_km']
                )
                
                if result is None:
                    os.remove(granule_path)
                    continue
                
                wls, arr_stack, target_lats, target_lons = result
            except Exception as e:
                logging.warning(f"Processing failed: {e}")
                try:
                    os.remove(granule_path)
                except:
                    pass
                continue
            
            # Extract datetime
            dt = extract_datetime_from_filename(filename)
            if dt is None:
                os.remove(granule_path)
                continue
            
            # Try to extract valid patches from this granule
            # Keep trying random locations until we get some valid patches or hit retry limit
            max_attempts_per_granule = 50  # Try up to 50 random locations per granule
            samples_from_this_granule = 0
            attempts = 0
            
            while samples_from_this_granule < 10 and attempts < max_attempts_per_granule and n_collected < n_to_collect:
                attempts += 1
                
                # Random location in Lake Erie
                lat = random.uniform(lat_min, lat_max)
                lon = random.uniform(lon_min, lon_max)
                
                # Round to grid precision to avoid resampling same pixel
                lat_rounded = round(lat, 4)
                lon_rounded = round(lon, 4)
                location_key = (lat_rounded, lon_rounded)
                
                # Skip if we've already sampled this location
                if location_key in sampled_locations[filename]:
                    continue
                
                sampled_locations[filename].add(location_key)
                
                # Extract patch
                patch_dict = extract_pace_patch(
                    arr_stack, wls, lon, lat,
                    pixel_count=patch_size,
                    lat_centers=target_lats,
                    lon_centers=target_lons
                )
                
                if patch_dict is None:
                    continue
                
                # Check if too many NaNs
                total_pixels = sum(arr.size for arr in patch_dict.values())
                valid_pixels = sum(np.count_nonzero(~np.isnan(arr)) for arr in patch_dict.values())
                valid_frac = valid_pixels / total_pixels if total_pixels > 0 else 0.0
                
                if valid_frac < config.MIN_VALID_FRACTION:
                    continue
                
                # Build feature vector
                patch_stack = np.stack(
                    [patch_dict[wl] for wl in sorted(patch_dict)],
                    axis=-1
                )
                global_means = np.nanmean(arr_stack, axis=(1, 2))
                feature_vector = np.concatenate([patch_stack.flatten(), global_means])
                
                # Create negative label
                t0 = pd.to_datetime(dt).tz_localize('UTC')
                label_tuple = (
                    f"BALANCED_{n_collected}",  # unique station name
                    t0,
                    lat,
                    lon,
                    0.0,  # PM concentration = 0 (negative)
                    np.nan,  # dissolved microcystin
                    np.nan   # chlorophyll-a
                )
                
                # Add to results
                results.append((
                    filename,
                    f"BALANCED_{n_collected}",
                    label_tuple,
                    feature_vector,
                    valid_frac,
                    patch_size
                ))
                
                n_collected += 1
                samples_from_this_granule += 1
                
                # Save incrementally every 10 samples
                if n_collected % 10 == 0:
                    np.save(training_path, np.array(results, dtype=object))
                    logging.info(f"Collected {n_collected}/{n_to_collect} negative samples")
            
            # Clean up granule file
            try:
                os.remove(granule_path)
            except:
                pass
            
            if samples_from_this_granule > 0:
                logging.info(f"  Extracted {samples_from_this_granule} valid samples from {filename}")
        
        if n_collected >= n_to_collect:
            break
        
        logging.info(f"Iteration {iteration} complete: {n_collected}/{n_to_collect} collected")
    
    # Final save of training data
    np.save(training_path, np.array(results, dtype=object))
    logging.info(f"Final collection complete: {n_collected}/{n_to_collect} negative samples collected")
    
    # ===== SAVE BALANCED DATA =====
    balanced_path = os.path.join(data_dir, f'training_data_balanced_{sensor}.npy')
    np.save(balanced_path, np.array(results, dtype=object))
    
    # Final distribution
    final_dist = analyze_class_distribution(results, pm_threshold)
    logging.info(f"\nFinal distribution:")
    logging.info(f"  Positive: {final_dist['positive']}")
    logging.info(f"  Negative: {final_dist['negative']}")
    logging.info(f"  Total: {final_dist['total']}")
    logging.info(f"  Imbalance ratio: {final_dist['imbalance_ratio']:.2f}")
    
    logging.info(f"Saved balanced data to {balanced_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance training data by adding negative samples')
    parser.add_argument('--sensor', type=str, default='PACE',
                        choices=['PACE', 'Sentinel-3'],
                        help='Sensor type')
    parser.add_argument('--patch-size', type=int, default=3,
                        choices=[3, 5, 7, 9],
                        help='Patch size')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='PM threshold for classification')
    parser.add_argument('--data-dir', type=str, default='./',
                        help='Data directory')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Maximum number of negative samples to collect (default: 2000)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for winter granules (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for winter granules (YYYY-MM-DD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    balance_by_oversampling_negatives(
        sensor=args.sensor,
        patch_size=args.patch_size,
        pm_threshold=args.threshold,
        data_dir=args.data_dir,
        n_negative_samples=args.n_samples,
        start_date=start_date,
        end_date=end_date,
        random_seed=args.seed
    )
