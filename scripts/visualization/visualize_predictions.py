"""
Generate prediction visualizations for PACE granules.

This script loads the trained model and generates spatial prediction maps
for selected PACE granules.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from microcystin_detection.predict import predict_from_granule
from microcystin_detection.model import load_model_with_normalization
from microcystin_detection import config


def plot_prediction_map(
    predictions: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    output_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0
):
    """
    Plot prediction map with Lake Erie context.
    
    Args:
        predictions: 2D array of probabilities
        lats: Latitude array
        lons: Longitude array
        title: Plot title
        output_path: Path to save figure
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create meshgrid for pcolormesh
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Plot prediction map
    cmap = plt.cm.RdYlGn_r  # Red for high risk, green for low
    im = ax.pcolormesh(
        lon_grid, lat_grid, predictions,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Microcystin Risk Probability', fontsize=12)
    
    # Add contours for high-risk areas
    contour_levels = [0.5, 0.7, 0.9]
    cs = ax.contour(
        lon_grid, lat_grid, predictions,
        levels=contour_levels,
        colors='black', linewidths=1.5, alpha=0.6
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt='%.1f')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction map to {output_path}")
    plt.close()


def main():
    """Generate predictions for sample granules."""
    
    # Setup paths
    model_path = config.BASE_DIR / 'model.keras'
    stats_dir = config.BASE_DIR
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model, norm_stats = load_model_with_normalization(str(model_path), str(stats_dir))
    
    if norm_stats is None:
        print("ERROR: Normalization stats not found")
        return
    
    print("Model loaded successfully!")
    
    # Get PACE parameters
    sensor_params = config.SENSOR_PARAMS['PACE']
    bbox = sensor_params['bbox']
    res_km = sensor_params['res_km']
    
    # Get wavelengths from a reference granule
    data_dir = Path('data')
    granule_files = sorted(data_dir.glob('PACE_OCI*.nc'))
    
    if not granule_files:
        print("ERROR: No PACE granules found in data/ directory")
        return
    
    # Get wavelengths from a reference granule
    import xarray as xr
    wavelengths = xr.open_dataset(
        str(granule_files[0]),
        group='sensor_band_parameters'
    )["wavelength_3d"].data
    
    print(f"Found {len(wavelengths)} wavelengths")
    print(f"Found {len(granule_files)} PACE granules")
    
    # Process a few selected granules from different months
    selected_granules = [
        'PACE_OCI.20240701T175112.L2.OC_AOP.V3_1.nc',  # July - bloom season
        'PACE_OCI.20240703T172214.L2.OC_AOP.V3_1.nc',  # July - bloom season
        'PACE_OCI.20240603T180158.L2.OC_AOP.V3_1.nc',  # June - early bloom
        'PACE_OCI.20240519T173132.L2.OC_AOP.V3_1.nc',  # May - pre-bloom
    ]
    
    for granule_name in selected_granules:
        granule_path = data_dir / granule_name
        
        if not granule_path.exists():
            print(f"SKIP: {granule_name} not found")
            continue
        
        print(f"\nProcessing {granule_name}...")
        
        # Extract date from filename
        date_str = granule_name.split('.')[1][:8]  # YYYYMMDD
        
        # Make predictions
        result = predict_from_granule(
            granule_path=str(granule_path),
            model=model,
            normalization_stats=norm_stats,
            patch_size=3,
            bbox=bbox,
            wavelengths=wavelengths,
            res_km=res_km,
            min_valid_frac=0.3
        )
        
        if result is None:
            print(f"  Failed to process {granule_name}")
            continue
        
        predictions, lats, lons = result
        
        # Statistics
        valid_mask = ~np.isnan(predictions)
        if valid_mask.sum() == 0:
            print(f"  No valid predictions for {granule_name}")
            continue
        
        valid_preds = predictions[valid_mask]
        mean_prob = np.mean(valid_preds)
        max_prob = np.max(valid_preds)
        high_risk = np.sum(valid_preds > 0.5)
        total_valid = valid_mask.sum()
        high_risk_pct = 100 * high_risk / total_valid
        
        print(f"  Valid pixels: {total_valid}")
        print(f"  Mean probability: {mean_prob:.3f}")
        print(f"  Max probability: {max_prob:.3f}")
        print(f"  High-risk pixels (>0.5): {high_risk} ({high_risk_pct:.1f}%)")
        
        # Plot
        title = f'Microcystin Risk Prediction - {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
        output_path = output_dir / f'prediction_{date_str}.png'
        
        plot_prediction_map(
            predictions, lats, lons,
            title, str(output_path)
        )
    
    print(f"\n✓ Visualization complete! Check the '{output_dir}' directory for outputs.")


if __name__ == '__main__':
    main()
