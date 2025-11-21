"""
Utility functions for PACE/Sentinel-3 data processing and patch extraction.

This module provides core functionality for:
- Processing satellite granules (PACE/Sentinel-3)
- Extracting patches around ground truth locations
- Regridding satellite data to uniform grids
- Visualization helpers
- Retry logic for remote data access
"""

import os
import re
import json
import math
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyresample import geometry, kd_tree
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd

from . import config


# ===== RETRY UTILITIES =====

def with_retries(
    fn: Callable,
    *args,
    max_retries: int = 20,
    retry_wait: Tuple[int, int] = (5, 10),
    **kwargs
) -> Any:
    """
    Run a function with exponential backoff retries on failure.
    
    Args:
        fn: The function to call
        *args: Positional arguments for fn
        max_retries: Maximum number of retry attempts
        retry_wait: Tuple (min_sec, max_sec) for random wait between retries
        **kwargs: Keyword arguments for fn
        
    Returns:
        Result of fn(*args, **kwargs) if successful
        
    Raises:
        Last exception raised by fn if all retries fail
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                wait_time = random.uniform(*retry_wait)
                logging.info(
                    f"Attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Attempt {attempt}/{max_retries} failed: {e}. No more retries."
                )
                raise


# ===== DATETIME UTILITIES =====

def extract_datetime_from_filename(filepath: str) -> Optional[datetime]:
    """
    Extract datetime from PACE/Sentinel filename.
    
    Expected format: YYYYMMDDTHHMMSS
    Example: PACE_OCI.20240517T163009.L2.OC.V2_0.NRT.nc
    
    Args:
        filepath: Path to satellite granule file
        
    Returns:
        datetime object or None if pattern not found
    """
    filename = os.path.basename(filepath)
    match = re.search(r"(\d{8}T\d{6})", filename)
    if not match:
        logging.warning(f"No timestamp pattern in filename: {filename}")
        return None
    
    timestamp_str = match.group(1)
    try:
        return datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
    except Exception as e:
        logging.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None


def estimate_position(
    times: List[pd.Timestamp],
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    target_time: pd.Timestamp
) -> Tuple[float, float]:
    """
    Estimate geographic position at target time from array of observations.
    
    Uses nearest-neighbor approach. For single observations, returns that location.
    
    Args:
        times: List of observation timestamps
        lat_arr: Latitude values (scalar or 1D array)
        lon_arr: Longitude values (scalar or 1D array)
        target_time: Target timestamp for position estimation
        
    Returns:
        Tuple of (latitude, longitude) at target_time
    """
    lat_arr = np.atleast_1d(lat_arr)
    lon_arr = np.atleast_1d(lon_arr)
    
    # Single observation case
    if lat_arr.size == 1 or lon_arr.size == 1:
        return float(lat_arr[0]), float(lon_arr[0])
    
    # Find nearest time
    deltas = [abs((ts - target_time).total_seconds()) for ts in times]
    nearest_idx = int(np.argmin(deltas))
    
    return float(lat_arr[nearest_idx]), float(lon_arr[nearest_idx])


# ===== PACE GRANULE PROCESSING =====

def process_pace_granule(
    filepath: str,
    bbox: Tuple[float, float, float, float],
    wavelengths: np.ndarray,
    res_km: float = 1.2
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process PACE granule: open, subset to bbox, regrid to uniform grid.
    
    Args:
        filepath: Path to PACE .nc file
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        wavelengths: Array of wavelengths to process (nm)
        res_km: Target resolution in kilometers
        
    Returns:
        Tuple of (wavelengths, arr_stack, target_lats, target_lons) or None if failed
        - wavelengths: Array of processed wavelengths
        - arr_stack: 3D array (n_wavelengths, n_lat, n_lon)
        - target_lats: 1D array of latitude centers
        - target_lons: 1D array of longitude centers
    """
    try:
        with xr.open_dataset(filepath, group="geophysical_data") as geo_ds, \
             xr.open_dataset(filepath, group="navigation_data") as nav_ds:
            
            # Merge navigation coordinates
            nav_ds = nav_ds.set_coords(("longitude", "latitude"))
            ds = xr.merge([geo_ds, nav_ds.coords])
            
            # Subset to bounding box
            ds = ds.where(
                (ds["latitude"] > bbox[1]) &
                (ds["latitude"] < bbox[3]) &
                (ds["longitude"] > bbox[0]) &
                (ds["longitude"] < bbox[2]),
                drop=True
            )
            
            # Get Rrs (remote sensing reflectance)
            rrs = ds["Rrs"]
            rrs = rrs.assign_coords(wavelength_3d=wavelengths)
            
            # Determine wavelength coordinate name
            if "wavelength_3d" in rrs.coords:
                wl_coord = "wavelength_3d"
            elif "wavelength" in rrs.coords:
                wl_coord = "wavelength"
            else:
                raise ValueError("Cannot find wavelength coordinate in Rrs")
            
            available_wavelengths = rrs[wl_coord].values
            available_wavelengths = np.atleast_1d(available_wavelengths)
            
    except Exception as e:
        logging.error(f"Failed to open PACE granule {filepath}: {e}")
        return None
    
    # Regrid each wavelength (with early termination if no bbox coverage)
    regridded_slices = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10  # Stop if 10 wavelengths in a row fail
    
    for i, wl in enumerate(available_wavelengths):
        try:
            slice_da = rrs.sel({wl_coord: wl}, method="nearest")
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.warning(f"Granule outside bbox - {consecutive_failures} consecutive wavelengths failed")
                break
            continue
        
        lat_arr = ds["latitude"].values
        lon_arr = ds["longitude"].values
        
        result = regrid_pace_slice(slice_da, lat_arr, lon_arr, bbox, res_km)
        if result is None:
            consecutive_failures += 1
            # Early termination: if first 10 wavelengths all fail, granule is outside bbox
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.warning(f"Granule outside bbox - {consecutive_failures} consecutive wavelengths failed")
                break
            continue
        
        # Success - reset counter
        consecutive_failures = 0
        regridded_2d, target_lats, target_lons = result
        regridded_slices.append((wl, regridded_2d))
    
    if not regridded_slices:
        logging.warning(f"No valid bands in {os.path.basename(filepath)}")
        return None
    
    # Stack into 3D array
    processed_wls, arrs = zip(*regridded_slices)
    arr_stack = np.stack(arrs, axis=0)
    
    return np.array(processed_wls), arr_stack, target_lats, target_lons


def regrid_pace_slice(
    slice_da: xr.DataArray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    res_km: float
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Regrid single wavelength slice to uniform lat/lon grid using nearest-neighbor.
    
    Args:
        slice_da: 2D DataArray of reflectance for one wavelength
        lat_arr: 2D latitude array from navigation_data
        lon_arr: 2D longitude array from navigation_data
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        res_km: Target resolution in kilometers
        
    Returns:
        Tuple of (regridded_2d, target_lats, target_lons) or None if failed
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Calculate resolution in degrees
    lat_center = (lat_min + lat_max) / 2.0
    res_lat_deg = res_km / 111.0  # km to degrees latitude
    res_lon_deg = res_km / (111.0 * math.cos(math.radians(lat_center)))
    
    # Build target grid
    target_lats = np.arange(lat_min, lat_max + 1e-6, res_lat_deg)
    target_lons = np.arange(lon_min, lon_max + 1e-6, res_lon_deg)
    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)
    
    # Flatten source data
    data = slice_da.values.flatten()
    lats = lat_arr.flatten()
    lons = lon_arr.flatten()
    
    # Apply spatial mask (bbox ± margin)
    mask = (
        (lons >= lon_min - 1.0) & (lons <= lon_max + 1.0) &
        (lats >= lat_min - 1.0) & (lats <= lat_max + 1.0)
    )
    
    data_local = data[mask]
    lats_local = lats[mask]
    lons_local = lons[mask]
    
    # Filter invalid values
    valid = np.isfinite(data_local) & np.isfinite(lats_local) & np.isfinite(lons_local)
    
    if not np.any(valid):
        # No logging here - let caller handle it to avoid log spam
        return None
    
    # Resample using pyresample
    swath_def = geometry.SwathDefinition(lons=lons_local[valid], lats=lats_local[valid])
    try:
        radius_m = res_km * 1000
        result = kd_tree.resample_nearest(
            swath_def,
            data_local[valid],
            area_def,
            radius_of_influence=radius_m,
            fill_value=np.nan
        )
    except Exception as e:
        logging.error(f"Resampling failed: {e}")
        return None
    
    return result, target_lats, target_lons


def extract_pace_patch(
    arr_stack: np.ndarray,
    wavelengths: np.ndarray,
    lon0: float,
    lat0: float,
    pixel_count: int,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray
) -> Optional[Dict[float, np.ndarray]]:
    """
    Extract square patch around (lon0, lat0) from regridded PACE data.
    
    Args:
        arr_stack: 3D array (n_wavelengths, n_lat, n_lon)
        wavelengths: 1D array of wavelengths (nm)
        lon0: Target longitude
        lat0: Target latitude
        pixel_count: Patch size (pixel_count × pixel_count)
        lat_centers: 1D array of grid latitude centers
        lon_centers: 1D array of grid longitude centers
        
    Returns:
        Dictionary mapping wavelength → 2D patch array, or None if extraction failed
    """
    # Find nearest grid cell
    lat_idx = np.abs(lat_centers - lat0).argmin()
    lon_idx = np.abs(lon_centers - lon0).argmin()
    
    half = pixel_count // 2
    ny, nx = arr_stack.shape[1], arr_stack.shape[2]
    
    patch_dict = {}
    for i, wl in enumerate(wavelengths):
        # Define slice with bounds checking
        i0 = max(0, lat_idx - half)
        i1 = min(ny, lat_idx + half + 1)
        j0 = max(0, lon_idx - half)
        j1 = min(nx, lon_idx + half + 1)
        
        patch = arr_stack[i, i0:i1, j0:j1]
        
        # Pad to exact size if needed
        if patch.shape != (pixel_count, pixel_count):
            pad_y = pixel_count - patch.shape[0]
            pad_x = pixel_count - patch.shape[1]
            pad_before_y = max(0, pad_y // 2)
            pad_before_x = max(0, pad_x // 2)
            pad_after_y = pad_y - pad_before_y
            pad_after_x = pad_x - pad_before_x
            
            patch = np.pad(
                patch,
                ((pad_before_y, pad_after_y), (pad_before_x, pad_after_x)),
                constant_values=np.nan
            )
        
        patch_dict[float(wl)] = patch
    
    return patch_dict


# ===== SENTINEL-3 PROCESSING =====

def regrid_granule(
    dataset: xr.Dataset,
    bbox: Tuple[float, float, float, float],
    res_km: float,
    chlor_a: bool = False
) -> Optional[Dict[str, xr.DataArray]]:
    """
    Regrid Sentinel-3 granule (Rrs bands or chlor_a) to uniform grid.
    
    Args:
        dataset: Opened Sentinel-3 xarray Dataset
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        res_km: Target resolution in kilometers
        chlor_a: If True, process chlor_a; if False, process Rrs_ bands
        
    Returns:
        Dictionary mapping band_name → DataArray on uniform grid, or None if failed
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Calculate resolution
    lat_center = (lat_min + lat_max) / 2.0
    res_lat_deg = res_km / 111.0
    res_lon_deg = res_km / (111.0 * math.cos(math.radians(lat_center)))
    
    # Build target grid
    target_lats = np.arange(lat_min, lat_max + 1e-6, res_lat_deg)
    target_lons = np.arange(lon_min, lon_max + 1e-6, res_lon_deg)
    lon2d, lat2d = np.meshgrid(target_lons, target_lats)
    area_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)
    
    # Extract coordinates
    lons = dataset["longitude"].values.flatten()
    lats = dataset["latitude"].values.flatten()
    
    # Spatial mask
    mask = (
        (lons >= lon_min - 1.0) & (lons <= lon_max + 1.0) &
        (lats >= lat_min - 1.0) & (lats <= lat_max + 1.0)
    )
    
    regridded = {}
    
    if not chlor_a:
        # Process Rrs bands
        bands = [name for name in dataset.data_vars if name.startswith("Rrs_")]
        if not bands:
            logging.warning("No Rrs_ bands found in granule")
            return None
        
        for band in bands:
            logging.info(f"Regridding band {band}")
            data = dataset[band].values.flatten()
            data[data == 0] = np.nan  # Treat zeros as missing
            
            data_local = data[mask]
            lons_local = lons[mask]
            lats_local = lats[mask]
            
            valid = ~np.isnan(data_local) & ~np.isnan(lons_local) & ~np.isnan(lats_local)
            
            if not np.any(valid):
                logging.warning(f"No valid data for {band}")
                return None
            
            swath_def = geometry.SwathDefinition(
                lons=lons_local[valid],
                lats=lats_local[valid]
            )
            
            try:
                radius_m = res_km * 1000
                result = kd_tree.resample_nearest(
                    swath_def,
                    data_local[valid],
                    area_def,
                    radius_of_influence=radius_m,
                    fill_value=np.nan
                )
            except Exception as e:
                logging.error(f"Resampling failed for {band}: {e}")
                return None
            
            da = xr.DataArray(
                result,
                dims=("latitude", "longitude"),
                coords={"latitude": target_lats, "longitude": target_lons},
                name=band
            )
            regridded[band] = da
    else:
        # Process chlor_a
        data = dataset["chlor_a"].values.flatten()
        data[data == 0] = np.nan
        
        data_local = data[mask]
        lons_local = lons[mask]
        lats_local = lats[mask]
        
        valid = ~np.isnan(data_local) & ~np.isnan(lons_local) & ~np.isnan(lats_local)
        
        if not np.any(valid):
            logging.warning("No valid chlor_a data")
            return None
        
        swath_def = geometry.SwathDefinition(
            lons=lons_local[valid],
            lats=lats_local[valid]
        )
        
        try:
            radius_m = res_km * 1000
            result = kd_tree.resample_nearest(
                swath_def,
                data_local[valid],
                area_def,
                radius_of_influence=radius_m,
                fill_value=np.nan
            )
        except Exception as e:
            logging.error(f"Resampling failed for chlor_a: {e}")
            return None
        
        da = xr.DataArray(
            result,
            dims=("latitude", "longitude"),
            coords={"latitude": target_lats, "longitude": target_lons},
            name="chlor_a"
        )
        regridded["chlor_a"] = da
    
    return regridded


def extract_patch_from_regridded(
    regridded: Dict[str, xr.DataArray],
    lon0: float,
    lat0: float,
    pixel_count: int,
    res_km: float,
    min_coverage: float = 0.4
) -> Optional[Tuple[Dict[str, np.ndarray], float, float]]:
    """
    Extract patch from regridded Sentinel-3 data with coverage check.
    
    Args:
        regridded: Dictionary of band_name → DataArray
        lon0: Target longitude
        lat0: Target latitude
        pixel_count: Patch size (pixel_count × pixel_count)
        res_km: Resolution in kilometers
        min_coverage: Minimum fraction of valid pixels required (default 0.4)
        
    Returns:
        Tuple of (patch_dict, half_lon_deg, half_lat_deg) or (None, None, None)
        - patch_dict: Dictionary mapping band_name → 2D patch array
        - half_lon_deg: Half-width of patch in longitude degrees
        - half_lat_deg: Half-width of patch in latitude degrees
    """
    half_km = (pixel_count * res_km) / 2.0
    half_lat_deg = half_km / 111.0
    half_lon_deg = half_km / (111.0 * math.cos(math.radians(lat0)))
    
    patch_arrays = {}
    total_cells = pixel_count * pixel_count
    
    for band, da in regridded.items():
        # Extract patch region
        patch = da.sel(
            longitude=slice(lon0 - half_lon_deg, lon0 + half_lon_deg),
            latitude=slice(lat0 - half_lat_deg, lat0 + half_lat_deg)
        ).values
        
        patch = patch.astype(float)
        patch[patch == 0] = np.nan
        
        # Adjust to exact size
        def adjust_size(arr: np.ndarray, target: int) -> np.ndarray:
            h, w = arr.shape
            if h < target or w < target:
                # Pad
                new = np.full((target, target), np.nan, dtype=arr.dtype)
                si = max((target - h) // 2, 0)
                sj = max((target - w) // 2, 0)
                new[si:si+h, sj:sj+w] = arr
                return new
            elif h > target or w > target:
                # Crop
                si = (h - target) // 2
                sj = (w - target) // 2
                return arr[si:si+target, sj:sj+target]
            else:
                return arr
        
        patch_adj = adjust_size(patch, pixel_count)
        
        # Check coverage
        valid_count = np.count_nonzero(~np.isnan(patch_adj))
        coverage = valid_count / total_cells
        
        if coverage < min_coverage:
            logging.info(
                f"Patch coverage for {band} is {coverage:.1%} < {min_coverage:.0%} → skip"
            )
            return None, None, None
        
        patch_arrays[band] = patch_adj
    
    return patch_arrays, half_lon_deg, half_lat_deg


# ===== VISUALIZATION =====

def stretch(
    array: np.ndarray,
    lower_percent: float = 1.0,
    upper_percent: float = 99.0
) -> np.ndarray:
    """
    Apply histogram stretching to enhance contrast.
    
    Args:
        array: Input array
        lower_percent: Lower percentile for clipping
        upper_percent: Upper percentile for clipping
        
    Returns:
        Stretched array clipped to [0, 1]
    """
    lower = np.nanpercentile(array, lower_percent)
    upper = np.nanpercentile(array, upper_percent)
    stretched = (array - lower) / (upper - lower)
    return np.clip(stretched, 0, 1)


def plot_granule(
    filepath: str,
    arr_stack: np.ndarray,
    bbox: Tuple[float, float, float, float],
    out_dir: str,
    rgb_bands: Tuple[int, int, int] = (105, 75, 48)
) -> None:
    """
    Create RGB visualization from PACE granule.
    
    Args:
        filepath: Path to original granule (for naming output)
        arr_stack: 3D array (n_wavelengths, n_lat, n_lon)
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        out_dir: Output directory for PNG
        rgb_bands: Tuple of (red_idx, green_idx, blue_idx) band indices
    """
    r_idx, g_idx, b_idx = rgb_bands
    
    # Extract and transpose bands
    r = arr_stack[r_idx, :, :].T
    g = arr_stack[g_idx, :, :].T
    b = arr_stack[b_idx, :, :].T
    
    # Stack and normalize
    rgb = np.dstack((r, g, b))
    rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    # Create figure with Cartopy
    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    # Basemap
    ax.set_facecolor("black")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor="none")
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="black", zorder=0)
    
    # Plot RGB
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    rgb_rot = np.transpose(rgb_norm, (1, 0, 2))
    
    ax.imshow(
        rgb_rot,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        zorder=2,
        interpolation="nearest"
    )
    
    # Add features
    ax.coastlines(resolution="10m")
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Save
    plt.tight_layout()
    granule_dir = os.path.join(out_dir, 'granules')
    os.makedirs(granule_dir, exist_ok=True)
    
    dt = extract_datetime_from_filename(filepath)
    if dt:
        filename = dt.strftime("%Y%m%d%H%M%S") + '.png'
    else:
        filename = 'unknown_' + os.path.basename(filepath).replace('.nc', '.png')
    
    output_path = os.path.join(granule_dir, filename)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    logging.info(f"Saved granule visualization to {output_path}")


def plot_true_color(
    ax: plt.Axes,
    regridded_dict: Dict[str, xr.DataArray],
    target_wavelengths: Dict[str, int] = None
) -> None:
    """
    Plot true-color RGB composite from Sentinel-3 Rrs bands.
    
    Args:
        ax: Matplotlib axis with Cartopy projection
        regridded_dict: Dictionary of band_name → DataArray
        target_wavelengths: Dict of {"red": 667, "green": 555, "blue": 443}
    """
    if target_wavelengths is None:
        target_wavelengths = {"red": 667, "green": 555, "blue": 443}
    
    # Extract wavelengths from band names
    band_names = list(regridded_dict.keys())
    wavelengths = {}
    for b in band_names:
        parts = b.split("_")
        try:
            wavelengths[b] = float(parts[-1])
        except (ValueError, IndexError):
            continue
    
    if len(wavelengths) < 3:
        logging.warning("Fewer than 3 Rrs bands; skipping true-color")
        return
    
    # Find closest bands to target wavelengths
    chosen = {}
    for color, target_wl in target_wavelengths.items():
        closest_band = min(wavelengths.keys(), key=lambda b: abs(wavelengths[b] - target_wl))
        chosen[color] = closest_band
    
    logging.info(f"True-color bands: {chosen}")
    
    # Extract arrays
    try:
        r = regridded_dict[chosen["red"]].values
        g = regridded_dict[chosen["green"]].values
        b = regridded_dict[chosen["blue"]].values
    except KeyError as e:
        logging.warning(f"Missing band {e}; aborting true-color")
        return
    
    # Stack and normalize
    rgb_raw = np.dstack((r, g, b)).astype(float)
    rgb_min = np.nanmin(rgb_raw)
    rgb_max = np.nanmax(rgb_raw)
    
    if rgb_max <= rgb_min:
        logging.warning("Zero dynamic range in RGB; skipping")
        return
    
    rgb_norm = (rgb_raw - rgb_min) / (rgb_max - rgb_min)
    
    # Create RGBA with alpha for NaN pixels
    h, w, _ = rgb_norm.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    rgba[..., :3] = rgb_norm
    mask_nan = np.isnan(rgb_norm).any(axis=2)
    rgba[..., 3] = np.where(mask_nan, 0.0, 1.0)
    
    # Get extent
    da0 = regridded_dict[chosen["red"]]
    lon0, lon1 = float(da0.longitude.min()), float(da0.longitude.max())
    lat0, lat1 = float(da0.latitude.min()), float(da0.latitude.max())
    extent = [lon0, lon1, lat0, lat1]
    
    # Plot
    ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="nearest"
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add features
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.5)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()


# ===== GEOMETRY UTILITIES =====

def shp_contains(
    geom,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray
) -> np.ndarray:
    """
    Check if grid points are inside shapely geometry.
    
    Args:
        geom: Shapely geometry object
        lon_grid: 2D array of longitudes
        lat_grid: 2D array of latitudes
        
    Returns:
        Boolean mask same shape as lon_grid/lat_grid
    """
    contains_vec = np.vectorize(lambda lon, lat: geom.contains(Point(lon, lat)))
    return contains_vec(lon_grid, lat_grid)


# ===== FEATURE EXTRACTION =====

def patch_to_features(patch_dict: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Convert patch dictionary to feature vector.
    
    Computes mean reflectance per wavelength, filtering NaNs.
    
    Args:
        patch_dict: Dictionary mapping wavelength → 2D patch array
        
    Returns:
        1D array of features (mean reflectance per wavelength)
    """
    patch_stack = np.stack(
        [patch_dict[wl] for wl in sorted(patch_dict)],
        axis=0
    )  # Shape: (n_wavelengths, patch_size, patch_size)
    
    flat = patch_stack.reshape(patch_stack.shape[0], -1)  # (n_wavelengths, patch_size²)
    features = np.nanmean(flat, axis=1)  # (n_wavelengths,)
    
    return features


# ===== LOGGING & CONFIGURATION =====

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the module.
    
    Args:
        level: Logging level (default INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_station_colors(json_path: str) -> Dict[str, str]:
    """Load station color mapping from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def save_station_colors(json_path: str, station_colors: Dict[str, str]) -> None:
    """Save station color mapping to JSON file."""
    with open(json_path, "w") as f:
        json.dump(station_colors, f, indent=2)


def assign_color_for_station(
    station_name: str,
    station_colors: Dict[str, str],
    color_cycle: List[str]
) -> str:
    """
    Assign consistent color to station name.
    
    Args:
        station_name: Name of station
        station_colors: Current mapping of station → color
        color_cycle: List of available colors
        
    Returns:
        Color string for this station
    """
    if station_name in station_colors:
        return station_colors[station_name]
    
    color = color_cycle[len(station_colors) % len(color_cycle)]
    station_colors[station_name] = color
    return color


def get_granule_filename(item: Any) -> Optional[str]:
    """
    Extract filename from earthaccess granule item.
    
    Args:
        item: earthaccess granule object
        
    Returns:
        Filename string or None if extraction failed
    """
    # Try common attribute names
    urls = None
    if hasattr(item, "data"):
        urls = item.data
    elif hasattr(item, "urls"):
        urls = item.urls
    elif hasattr(item, "get_data_urls"):
        try:
            urls = item.get_data_urls()
        except:
            urls = None
    
    # Fallback to parsing repr
    if urls is None:
        txt = repr(item)
        match = re.search(r"Data:\s*\[\s*['\"](https?://[^'\"]+)['\"]", txt)
        if match:
            urls = [match.group(1)]
    
    if not urls:
        return None
    
    # Extract filename from first URL
    url = urls[0]
    return os.path.basename(url)
