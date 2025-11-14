"""
Prediction module for microcystin detection.

This module handles:
- Loading trained CNN models
- Making predictions on new satellite data
- Generating spatial prediction maps
- Ensemble predictions from multiple models
"""

import os
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

from . import config
from .utils import (
    process_pace_granule,
    extract_pace_patch,
    extract_datetime_from_filename,
    configure_logging
)
from .model import load_model_with_normalization


def normalize_patch(
    patch: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Normalize patch using saved statistics.
    
    Args:
        patch: Patch array (patch_size, patch_size, n_channels + 1)
               Last channel is mask
        means: Channel means
        stds: Channel standard deviations
        
    Returns:
        Normalized patch
    """
    n_channels = len(means)
    
    # Normalize spectral channels (excluding mask)
    patch[..., :n_channels] = (patch[..., :n_channels] - means) / (stds + 1e-6)
    patch = np.nan_to_num(patch, nan=0.0)
    
    return patch


def normalize_context(
    context: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Normalize context features using saved statistics.
    
    Args:
        context: Context feature vector
        means: Context feature means
        stds: Context feature standard deviations
        
    Returns:
        Normalized context
    """
    context = (context - means) / (stds + 1e-6)
    context = np.nan_to_num(context, nan=0.0)
    
    return context


def predict_from_granule(
    granule_path: str,
    model: tf.keras.Model,
    normalization_stats: Dict[str, np.ndarray],
    patch_size: int,
    bbox: Tuple[float, float, float, float],
    wavelengths: np.ndarray,
    res_km: float = 1.2,
    min_valid_frac: float = 0.5
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Make predictions on a single satellite granule.
    
    Args:
        granule_path: Path to satellite granule file
        model: Loaded Keras model
        normalization_stats: Dict with 'patch_means', 'patch_stds', 
                            'context_means', 'context_stds'
        patch_size: Patch dimension
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        wavelengths: Array of wavelengths
        res_km: Resolution in kilometers
        min_valid_frac: Minimum valid pixel fraction
        
    Returns:
        Tuple of (predictions, lats, lons) or None if processing failed
        - predictions: 2D array of probabilities
        - lats: 1D array of latitudes
        - lons: 1D array of longitudes
    """
    # Process granule
    result = process_pace_granule(granule_path, bbox, wavelengths, res_km)
    if result is None:
        logging.warning(f"Failed to process {os.path.basename(granule_path)}")
        return None
    
    wls, arr_stack, target_lats, target_lons = result
    
    # Compute global means for context features
    global_means = np.nanmean(arr_stack, axis=(1, 2))
    
    # Get normalization stats
    patch_means = normalization_stats['patch_means']
    patch_stds = normalization_stats['patch_stds']
    context_means = normalization_stats['context_means']
    context_stds = normalization_stats['context_stds']
    
    # Create prediction grid
    n_lat, n_lon = len(target_lats), len(target_lons)
    predictions = np.full((n_lat, n_lon), np.nan)
    
    # Loop over grid points
    for i, lat in enumerate(target_lats):
        for j, lon in enumerate(target_lons):
            # Extract patch
            patch_dict = extract_pace_patch(
                arr_stack, wls, lon, lat,
                pixel_count=patch_size,
                lat_centers=target_lats,
                lon_centers=target_lons
            )
            
            if patch_dict is None:
                continue
            
            # Check valid fraction
            total = sum(arr.size for arr in patch_dict.values())
            valid = sum(np.count_nonzero(~np.isnan(arr)) for arr in patch_dict.values())
            valid_frac = valid / total if total > 0 else 0.0
            
            if valid_frac < min_valid_frac:
                continue
            
            # Build patch array
            patch_stack = np.stack(
                [patch_dict[wl] for wl in sorted(patch_dict)],
                axis=-1
            )  # (patch_size, patch_size, n_channels)
            
            # Add mask channel
            mask = np.any(~np.isnan(patch_stack), axis=-1, keepdims=True).astype('float32')
            patch = np.concatenate([patch_stack, mask], axis=-1)
            
            # Normalize patch
            patch = normalize_patch(patch, patch_means, patch_stds)
            
            # Build context features: [global_means]
            context = global_means.copy()
            context = normalize_context(context, context_means, context_stds)
            
            # Predict
            patch_input = np.expand_dims(patch, axis=0)  # (1, P, P, C+1)
            context_input = np.expand_dims(context, axis=0)  # (1, C)
            
            pred = model.predict([patch_input, context_input], verbose=0)[0, 0]
            predictions[i, j] = pred
    
    return predictions, target_lats, target_lons


def ensemble_predict(
    granule_path: str,
    model_dirs: List[str],
    patch_size: int,
    bbox: Tuple[float, float, float, float],
    wavelengths: np.ndarray,
    res_km: float = 1.2,
    min_valid_frac: float = 0.5,
    prob_threshold: float = 0.5
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Ensemble predictions from multiple trained models.
    
    Args:
        granule_path: Path to satellite granule
        model_dirs: List of directories containing model.keras and stats
        patch_size: Patch dimension
        bbox: Bounding box
        wavelengths: Wavelengths to process
        res_km: Resolution
        min_valid_frac: Minimum valid pixel fraction
        prob_threshold: Threshold for binary classification
        
    Returns:
        Tuple of (ensemble_predictions, lats, lons) or None
        - ensemble_predictions: 2D array of ensemble probabilities (0-1)
        - lats: 1D latitude array
        - lons: 1D longitude array
    """
    all_predictions = []
    lats, lons = None, None
    
    for model_dir in model_dirs:
        # Load model and stats
        model_path = os.path.join(model_dir, 'model.keras')
        
        if not os.path.exists(model_path):
            logging.warning(f"Model not found: {model_path}")
            continue
        
        model, stats = load_model_with_normalization(model_path, model_dir)
        
        if stats is None:
            logging.warning(f"Missing normalization stats for {model_dir}")
            continue
        
        # Make predictions
        result = predict_from_granule(
            granule_path, model, stats, patch_size,
            bbox, wavelengths, res_km, min_valid_frac
        )
        
        if result is None:
            continue
        
        preds, lats, lons = result
        all_predictions.append(preds)
    
    if not all_predictions:
        logging.warning("No valid predictions from any model")
        return None
    
    # Ensemble: average predictions
    ensemble = np.nanmean(all_predictions, axis=0)
    
    return ensemble, lats, lons


def predict_time_series(
    start_date: date,
    end_date: date,
    model_path: str,
    stats_dir: str,
    patch_size: int,
    sensor: str = 'PACE',
    output_dir: str = './predictions',
    days_lookback: int = 7,
    min_valid_frac: float = 0.5
) -> pd.DataFrame:
    """
    Generate predictions for a time series of dates.
    
    Args:
        start_date: Start date
        end_date: End date
        model_path: Path to trained model
        stats_dir: Directory with normalization stats
        patch_size: Patch dimension
        sensor: Sensor name
        output_dir: Output directory for prediction maps
        days_lookback: Number of days to look back for data
        min_valid_frac: Minimum valid pixel fraction
        
    Returns:
        DataFrame with columns: date, mean_prob, max_prob, high_risk_pixels, total_pixels
    """
    configure_logging()
    
    # Load model
    model, stats = load_model_with_normalization(model_path, stats_dir)
    if stats is None:
        logging.error("Could not load normalization statistics")
        return pd.DataFrame()
    
    # Get sensor params
    sensor_params = config.SENSOR_PARAMS[sensor.upper()]
    bbox = sensor_params['bbox']
    res_km = sensor_params['res_km']
    
    # Get wavelengths
    logging.info("Loading wavelengths...")
    import earthaccess
    auth = earthaccess.login(persist=True)
    
    from .utils import with_retries
    ref_search = with_retries(
        earthaccess.search_data,
        short_name="PACE_OCI_L2_AOP",
        temporal=("2024-06-01", "2024-06-05"),
        bounding_box=bbox
    )
    ref_file = with_retries(earthaccess.download, ref_search, "./data")[0]
    wavelengths = xr.open_dataset(
        ref_file,
        group="sensor_band_parameters"
    )["wavelength_3d"].data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Loop over dates
    current_date = start_date
    while current_date <= end_date:
        logging.info(f"Processing {current_date}")
        
        # Search for granules
        date_start = current_date - timedelta(days=days_lookback)
        date_end = current_date + timedelta(days=1)
        
        start_iso = date_start.strftime("%Y-%m-%dT00:00:00Z")
        end_iso = date_end.strftime("%Y-%m-%dT00:00:00Z")
        
        granules = []
        for short_name in sensor_params['short_names']:
            try:
                search_results = with_retries(
                    earthaccess.search_data,
                    short_name=short_name,
                    temporal=(start_iso, end_iso),
                    bounding_box=bbox
                )
                if search_results:
                    granules.extend(search_results)
            except Exception as e:
                logging.error(f"Search failed: {e}")
        
        if not granules:
            logging.warning(f"No granules found for {current_date}")
            current_date += timedelta(days=1)
            continue
        
        # Download and process granules
        try:
            paths = with_retries(earthaccess.download, granules, './data')
        except Exception as e:
            logging.error(f"Download failed: {e}")
            current_date += timedelta(days=1)
            continue
        
        # Make predictions for each granule and aggregate
        all_preds = []
        for path in paths:
            result = predict_from_granule(
                path, model, stats, patch_size,
                bbox, wavelengths, res_km, min_valid_frac
            )
            
            if result is not None:
                preds, lats, lons = result
                all_preds.append(preds)
        
        if not all_preds:
            logging.warning(f"No valid predictions for {current_date}")
            current_date += timedelta(days=1)
            continue
        
        # Aggregate predictions
        daily_pred = np.nanmean(all_preds, axis=0)
        
        # Compute statistics
        valid_mask = ~np.isnan(daily_pred)
        if valid_mask.sum() > 0:
            mean_prob = np.nanmean(daily_pred)
            max_prob = np.nanmax(daily_pred)
            high_risk = (daily_pred >= 0.5).sum()
            total = valid_mask.sum()
        else:
            mean_prob = np.nan
            max_prob = np.nan
            high_risk = 0
            total = 0
        
        results.append({
            'date': current_date,
            'mean_probability': mean_prob,
            'max_probability': max_prob,
            'high_risk_pixels': high_risk,
            'total_pixels': total,
            'high_risk_fraction': high_risk / total if total > 0 else 0
        })
        
        # Save prediction map
        pred_path = os.path.join(output_dir, f'prediction_{current_date}.npy')
        np.save(pred_path, daily_pred)
        logging.info(f"Saved prediction to {pred_path}")
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'prediction_summary.csv')
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Saved summary to {csv_path}")
    
    return df_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate microcystin predictions')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.keras file)')
    parser.add_argument('--stats-dir', type=str, required=True,
                        help='Directory with normalization statistics')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--patch-size', type=int, default=3,
                        help='Patch size')
    parser.add_argument('--sensor', type=str, default='PACE',
                        help='Sensor name')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Output directory')
    parser.add_argument('--lookback', type=int, default=7,
                        help='Days to look back for data')
    parser.add_argument('--min-valid', type=float, default=0.5,
                        help='Minimum valid pixel fraction')
    
    args = parser.parse_args()
    
    # Parse dates
    start = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Run predictions
    results = predict_time_series(
        start_date=start,
        end_date=end,
        model_path=args.model_path,
        stats_dir=args.stats_dir,
        patch_size=args.patch_size,
        sensor=args.sensor,
        output_dir=args.output_dir,
        days_lookback=args.lookback,
        min_valid_frac=args.min_valid
    )
    
    print(f"\nPrediction Summary:")
    print(results)
