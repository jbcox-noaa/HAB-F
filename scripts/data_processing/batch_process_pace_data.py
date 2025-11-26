#!/usr/bin/env python3
"""
Batch process PACE data month-by-month to save disk space.

Strategy:
1. Download one month of PACE granules
2. Generate MC probability maps for that month
3. Delete the .nc files
4. Repeat for next month

This allows processing all available data without requiring 77GB of storage.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

import earthaccess
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from microcystin_detection.predict import ensemble_predict

# Configuration
BBOX = (-83.5, 41.3, -82.45, 42.2)  # Lake Erie
DATA_DIR = Path("./data")
TEMP_DIR = Path("./data/temp_monthly")
OUTPUT_DIR = Path("./data/MC_probability_maps")
SHORT_NAMES = ["PACE_OCI_L2_AOP"]
RES_KM = 1.2
MIN_VALID_FRAC = 0.2

# Model ensemble paths
MC_MODEL_DIRS = [
    Path("./microcystin_detection/models/patch_3"),
    Path("./microcystin_detection/models/patch_5"),
    Path("./microcystin_detection/models/patch_7"),
    Path("./microcystin_detection/models/patch_9"),
]
PATCH_SIZES = [3, 5, 7, 9]


def get_month_ranges() -> List[Tuple[str, str]]:
    """
    Generate list of (start_date, end_date) tuples for each month.
    
    Returns:
        List of month ranges from PACE launch through current
    """
    # PACE launched Feb 2024, get data through Nov 2025 (5 days buffer)
    start = datetime(2024, 2, 1).date()
    end = (datetime.now() - timedelta(days=5)).date()
    
    months = []
    current = start
    
    while current <= end:
        # Start of month
        month_start = current.strftime("%Y-%m-%d")
        
        # End of month (or end date if in final month)
        current_dt = datetime(current.year, current.month, 28)
        next_month = current_dt + timedelta(days=4)
        month_end = (next_month.replace(day=1) - timedelta(days=1)).date()
        
        if month_end > end:
            month_end = end
        
        months.append((month_start, month_end.strftime("%Y-%m-%d")))
        
        # Move to next month
        next_month_dt = datetime(current.year, current.month, 1) + timedelta(days=32)
        current = next_month_dt.replace(day=1).date()
        
        if current > end:
            break
    
    return months


def download_month(start_date: str, end_date: str) -> List[str]:
    """
    Download PACE granules for a specific month.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of downloaded file paths
    """
    logger.info(f"Downloading PACE data for {start_date} to {end_date}...")
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Authenticate
    auth = earthaccess.login()
    if not auth.authenticated:
        logger.error("Authentication failed!")
        return []
    
    # Search for granules
    try:
        results = earthaccess.search_data(
            short_name=SHORT_NAMES,
            bounding_box=BBOX,
            temporal=(start_date, end_date),
            count=-1
        )
        
        logger.info(f"Found {len(results)} granules")
        
        if len(results) == 0:
            logger.warning("No granules found for this period")
            return []
        
        # Download to temp directory
        paths = earthaccess.download(results, str(TEMP_DIR))
        logger.info(f"Downloaded {len(paths)} granules ({sum(os.path.getsize(p) for p in paths) / 1e9:.2f} GB)")
        
        return paths
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return []


def process_month_granules(granule_paths: List[str], start_date: str) -> int:
    """
    Generate MC probability maps for downloaded granules.
    
    Args:
        granule_paths: List of granule file paths
        start_date: Month identifier for logging
        
    Returns:
        Number of maps successfully generated
    """
    logger.info(f"Processing {len(granule_paths)} granules from {start_date}...")
    
    # Load wavelengths from first granule
    import xarray as xr
    ref_granule = granule_paths[0]
    with xr.open_dataset(ref_granule, group="sensor_band_parameters") as ds:
        wavelengths = ds["wavelength_3d"].values
    logger.info(f"Loaded {len(wavelengths)} wavelength bands")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Group granules by date
    from collections import defaultdict
    date_groups = defaultdict(list)
    
    for path in granule_paths:
        filename = os.path.basename(path)
        # Extract date: PACE_OCI.YYYYMMDDTHHMMSS...
        try:
            date_str = filename.split('.')[1][:8]  # YYYYMMDD
            date_groups[date_str].append(path)
        except:
            logger.warning(f"Could not parse date from {filename}")
            continue
    
    logger.info(f"Found {len(date_groups)} unique dates")
    
    # Process each date
    maps_generated = 0
    
    for date_str in sorted(date_groups.keys()):
        granule_list = date_groups[date_str]
        output_file = OUTPUT_DIR / f"mc_probability_{date_str}.npy"
        
        # Skip if already exists
        if output_file.exists():
            logger.info(f"[{date_str}] Already processed - skipping")
            maps_generated += 1
            continue
        
        logger.info(f"[{date_str}] Processing {len(granule_list)} granule(s)")
        
        daily_predictions = []
        daily_lats = []
        daily_lons = []
        
        for granule_path in granule_list:
            filename = os.path.basename(granule_path)
            
            try:
                # Run ensemble prediction
                patch_predictions = []
                lats, lons = None, None
                
                for model_dir, patch_size in zip(MC_MODEL_DIRS, PATCH_SIZES):
                    result = ensemble_predict(
                        granule_path=granule_path,
                        model_dirs=[str(model_dir)],
                        patch_size=patch_size,
                        bbox=BBOX,
                        wavelengths=wavelengths,
                        res_km=RES_KM,
                        min_valid_frac=MIN_VALID_FRAC
                    )
                    
                    if result is not None:
                        preds, lats, lons = result
                        patch_predictions.append(preds)
                        logger.debug(f"    Patch {patch_size}: {np.sum(~np.isnan(preds))} valid pixels")
                
                if not patch_predictions:
                    logger.warning(f"  {filename}: No valid predictions")
                    continue
                
                # Average across patch sizes
                predictions = np.nanmean(patch_predictions, axis=0)
                valid_pixels = np.sum(~np.isnan(predictions))
                
                daily_predictions.append(predictions)
                daily_lats.append(lats)
                daily_lons.append(lons)
                
                logger.info(f"  {filename}: SUCCESS ({valid_pixels} valid pixels)")
                
            except Exception as e:
                logger.error(f"  {filename}: ERROR - {e}")
                continue
        
        # Save daily map if we got any predictions
        if daily_predictions:
            # Use first granule's grid (they should all be the same)
            final_predictions = daily_predictions[0]
            
            # Average if multiple granules per day
            if len(daily_predictions) > 1:
                final_predictions = np.nanmean(daily_predictions, axis=0)
            
            # Save as .npy
            np.save(output_file, final_predictions)
            
            # Save coordinates
            coords_file = OUTPUT_DIR / f"mc_probability_{date_str}_coords.npz"
            np.savez(coords_file, lats=daily_lats[0], lons=daily_lons[0])
            
            logger.info(f"[{date_str}] ✅ Saved map with {np.sum(~np.isnan(final_predictions))} valid pixels")
            maps_generated += 1
        else:
            logger.warning(f"[{date_str}] No valid predictions - skipping")
    
    return maps_generated


def cleanup_temp_files():
    """Delete temporary .nc files."""
    if TEMP_DIR.exists():
        size_before = sum(f.stat().st_size for f in TEMP_DIR.glob('*.nc')) / 1e9
        logger.info(f"Deleting {size_before:.2f} GB of temporary .nc files...")
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Cleanup complete")


def main():
    """Main batch processing workflow."""
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING: PACE DATA → MC PROBABILITY MAPS")
    logger.info("=" * 70)
    
    # Get month ranges
    months = get_month_ranges()
    logger.info(f"Will process {len(months)} months of data")
    logger.info(f"Date range: {months[0][0]} to {months[-1][1]}")
    
    total_maps = 0
    
    for i, (start_date, end_date) in enumerate(months, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MONTH {i}/{len(months)}: {start_date} to {end_date}")
        logger.info("=" * 70)
        
        # Download month
        granule_paths = download_month(start_date, end_date)
        
        if not granule_paths:
            logger.warning(f"No data for {start_date}, skipping...")
            continue
        
        # Process month
        maps_generated = process_month_granules(granule_paths, start_date)
        total_maps += maps_generated
        
        # Cleanup
        cleanup_temp_files()
        
        logger.info(f"Month complete: {maps_generated} maps generated")
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total MC probability maps generated: {total_maps}")
    logger.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
