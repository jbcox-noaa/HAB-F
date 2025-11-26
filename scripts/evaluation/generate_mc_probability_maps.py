#!/usr/bin/env python3
"""
Generate Microcystin Probability Maps from PACE Granules (Phase 4)

This script processes all available PACE L2 AOP granules through a 4-model
ensemble of microcystin detection CNN classifiers to create daily MC probability maps.
These maps will be used as input for Phase 4 microcystin forecasting.

Ensemble Configuration:
    - 4 CNN models with different patch sizes (3x3, 5x5, 7x7, 9x9)
    - Each model trained with temporal split to prevent data leakage
    - Predictions averaged across all models for robust probability estimates
    - Expected realistic probabilities (not 100%) due to proper validation

Output:
    - Daily MC probability maps saved as .npy files in data/MC_probability_maps/
    - Each file contains a 2D array of probabilities [0, 1] for MC ≥ 1.0 µg/L
    - Spatial grid consistent across all dates
    - Metadata saved in JSON format with ensemble configuration
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple, List
import numpy as np
import tensorflow as tf
from glob import glob
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from microcystin_detection.predict import predict_from_granule, ensemble_predict
from microcystin_detection.utils import process_pace_granule
import microcystin_detection.config as mc_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mc_prob_map_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PACE_DATA_DIR = DATA_DIR  # PACE granules in data/
OUTPUT_DIR = DATA_DIR / "MC_probability_maps"

# Microcystin detection model ensemble (Phase 4)
# Using 4 models with different patch sizes for robust predictions
MC_MODEL_DIRS = [
    BASE_DIR / "microcystin_detection" / "models" / "patch_3",
    BASE_DIR / "microcystin_detection" / "models" / "patch_5",
    BASE_DIR / "microcystin_detection" / "models" / "patch_7",
    BASE_DIR / "microcystin_detection" / "models" / "patch_9",
]
PATCH_SIZES = [3, 5, 7, 9]  # Corresponding patch sizes

# PACE parameters
BBOX = (-83.5, 41.3, -82.45, 42.2)  # Lake Erie
RES_KM = 1.2  # Spatial resolution
MIN_VALID_FRAC = 0.2  # Minimum valid pixel fraction per patch (lowered for sparse coverage)

# Quality control
# Note: No minimum pixel threshold - save all granules with ANY valid predictions
# Each pixel prediction is independent, so even partial coverage is valuable

# Processing
OVERWRITE = False  # Set to True to regenerate existing maps

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date string from PACE granule filename.
    
    Example:
        PACE_OCI.20240506T180930.L2.OC_AOP.V3_1.nc -> 20240506
    
    Returns:
        Date string in YYYYMMDD format or None if parsing fails
    """
    try:
        # Format: PACE_OCI.YYYYMMDDTHHMMSS.L2.OC_AOP.V3_1.nc
        parts = filename.split('.')
        datetime_str = parts[1]  # 20240506T180930
        date_str = datetime_str.split('T')[0]  # 20240506
        return date_str
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse date from {filename}: {e}")
        return None


def aggregate_daily_predictions(
    predictions_list: List[np.ndarray],
    lats_list: List[np.ndarray],
    lons_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate multiple predictions for the same day into a single map.
    
    For days with multiple granules (overlapping passes), we take the mean
    of predictions at each location.
    
    Args:
        predictions_list: List of 2D prediction arrays
        lats_list: List of latitude arrays
        lons_list: List of longitude arrays
        
    Returns:
        Tuple of (aggregated_predictions, lats, lons)
    """
    if len(predictions_list) == 1:
        return predictions_list[0], lats_list[0], lons_list[0]
    
    # Check if all grids are identical
    lats = lats_list[0]
    lons = lons_list[0]
    for i in range(1, len(lats_list)):
        if not (np.allclose(lats_list[i], lats) and np.allclose(lons_list[i], lons)):
            logger.warning("Grid mismatch detected - using first grid")
            break
    
    # Stack predictions and take mean (ignoring NaN)
    pred_stack = np.stack(predictions_list, axis=0)
    aggregated = np.nanmean(pred_stack, axis=0)
    
    logger.info(f"Aggregated {len(predictions_list)} granules for this date")
    
    return aggregated, lats, lons


def load_normalization_stats() -> dict:
    """
    Load normalization statistics for Phase 2 model.
    
    NOTE: This function is deprecated in favor of using ensemble_predict()
    which loads stats from each model directory automatically.
    """
    logger.warning("load_normalization_stats() is deprecated - using ensemble_predict() instead")
    return {}


def load_wavelengths() -> np.ndarray:
    """Load PACE wavelengths from a reference granule."""
    # Find first available granule to extract wavelengths
    pace_files = sorted(glob(str(PACE_DATA_DIR / "PACE_OCI*.nc")))
    if not pace_files:
        raise FileNotFoundError("No PACE granules found")
    
    import xarray as xr
    ref_file = pace_files[0]
    logger.info(f"Loading wavelengths from: {os.path.basename(ref_file)}")
    
    with xr.open_dataset(ref_file, group="sensor_band_parameters") as ds:
        wavelengths = ds["wavelength_3d"].values
    
    return wavelengths


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def generate_mc_probability_maps():
    """
    Main function to generate MC probability maps from all PACE granules.
    """
    logger.info("="*70)
    logger.info("GENERATING MICROCYSTIN PROBABILITY MAPS FROM PACE DATA")
    logger.info("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all PACE granules
    pace_files = sorted(glob(str(PACE_DATA_DIR / "PACE_OCI*.nc")))
    logger.info(f"Found {len(pace_files)} PACE granule files")
    
    if not pace_files:
        logger.error("No PACE granules found - exiting")
        return
    
    # Load wavelengths
    logger.info("Loading wavelengths...")
    wavelengths = load_wavelengths()
    logger.info(f"Loaded {len(wavelengths)} wavelength bands")
    
    # Verify all ensemble models exist
    logger.info("Verifying ensemble models...")
    logger.info(f"Ensemble configuration: {len(MC_MODEL_DIRS)} models with patch sizes {PATCH_SIZES}")
    
    missing_models = []
    for model_dir, patch_size in zip(MC_MODEL_DIRS, PATCH_SIZES):
        model_path = model_dir / "model.keras"
        if not model_path.exists():
            missing_models.append((patch_size, str(model_path)))
            logger.error(f"  Patch {patch_size}: Model NOT found at {model_path}")
        else:
            logger.info(f"  Patch {patch_size}: Model found ✓")
    
    if missing_models:
        logger.error(f"Missing {len(missing_models)} models - please train them first:")
        for patch_size, path in missing_models:
            logger.error(f"  - Patch {patch_size}: {path}")
        return
    
    logger.info("All ensemble models verified successfully!")
    
    # Group granules by date
    date_groups = {}
    for granule_path in pace_files:
        filename = os.path.basename(granule_path)
        date_str = extract_date_from_filename(filename)
        if date_str is None:
            logger.warning(f"Skipping {filename} - could not extract date")
            continue
        
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(granule_path)
    
    logger.info(f"Found {len(date_groups)} unique dates")
    
    # Process each date
    processed = 0
    skipped = 0
    failed = 0
    
    for date_str in sorted(date_groups.keys()):
        granule_paths = date_groups[date_str]
        output_path = OUTPUT_DIR / f"{date_str}_mc_prob.npy"
        
        # Check if already processed
        if output_path.exists() and not OVERWRITE:
            logger.info(f"[{date_str}] Already processed - skipping")
            skipped += 1
            continue
        
        logger.info(f"[{date_str}] Processing {len(granule_paths)} granule(s)")
        
        # Process each granule for this date
        daily_predictions = []
        daily_lats = []
        daily_lons = []
        
        for granule_path in granule_paths:
            filename = os.path.basename(granule_path)
            
            try:
                # Run Phase 4 ensemble prediction with all 4 models
                # This automatically loads each model and its normalization stats
                logger.info(f"  {filename}: Running ensemble prediction with {len(MC_MODEL_DIRS)} models...")
                
                # Try each patch size and aggregate
                patch_predictions = []
                lats, lons = None, None
                
                for model_dir, patch_size in zip(MC_MODEL_DIRS, PATCH_SIZES):
                    result = ensemble_predict(
                        granule_path=granule_path,
                        model_dirs=[str(model_dir)],  # Single model for this patch size
                        patch_size=patch_size,
                        bbox=BBOX,
                        wavelengths=wavelengths,
                        res_km=RES_KM,
                        min_valid_frac=MIN_VALID_FRAC
                    )
                    
                    if result is not None:
                        preds, lats, lons = result
                        patch_predictions.append(preds)
                        logger.info(f"    Patch {patch_size}: {np.sum(~np.isnan(preds))} valid pixels")
                
                if not patch_predictions:
                    logger.warning(f"  {filename}: No valid predictions from any model")
                    continue
                
                # Ensemble: average across all patch sizes
                predictions = np.nanmean(patch_predictions, axis=0)
                
                # Count valid pixels for logging
                valid_pixels = np.sum(~np.isnan(predictions))
                
                daily_predictions.append(predictions)
                daily_lats.append(lats)
                daily_lons.append(lons)
                
                logger.info(
                    f"  {filename}: SUCCESS "
                    f"({valid_pixels} valid pixels, "
                    f"shape: {predictions.shape})"
                )
                
            except Exception as e:
                logger.error(f"  {filename}: ERROR - {e}")
                continue
        
        # Aggregate daily predictions
        if not daily_predictions:
            logger.warning(f"[{date_str}] No valid predictions - skipping date")
            failed += 1
            continue
        
        mc_prob_map, lats, lons = aggregate_daily_predictions(
            daily_predictions, daily_lats, daily_lons
        )
        
        # Save MC probability map
        np.save(output_path, mc_prob_map)
        logger.info(
            f"[{date_str}] ✅ SAVED: {output_path.name} "
            f"(shape: {mc_prob_map.shape}, "
            f"valid: {np.sum(~np.isnan(mc_prob_map))} pixels)"
        )
        
        processed += 1
    
    # Summary
    logger.info("="*70)
    logger.info("GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total dates: {len(date_groups)}")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped (already exist): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Save metadata
    metadata = {
        'total_dates': len(date_groups),
        'processed': processed,
        'skipped': skipped,
        'failed': failed,
        'output_dir': str(OUTPUT_DIR),
        'bbox': BBOX,
        'res_km': RES_KM,
        'min_valid_frac_per_patch': MIN_VALID_FRAC,  # Per-patch threshold
        'ensemble_config': {
            'n_models': len(MC_MODEL_DIRS),
            'patch_sizes': PATCH_SIZES,
            'model_dirs': [str(d) for d in MC_MODEL_DIRS]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    generate_mc_probability_maps()
