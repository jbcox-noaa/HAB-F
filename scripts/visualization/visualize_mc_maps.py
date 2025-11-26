#!/usr/bin/env python3
"""
Visualize MC Probability Maps
Displays microcystin probability predictions for Lake Erie from PACE satellite data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep
import cartopy.io.shapereader as shpreader

# Configuration
DATA_DIR = Path("./data/MC_probability_maps")
BBOX = (-83.5, 41.3, -82.45, 42.2)  # Lake Erie (lon_min, lat_min, lon_max, lat_max)

# Cache for lake geometry
_LAKE_GEOMETRY = None

def get_lake_erie_geometry():
    """Get the Lake Erie polygon from Natural Earth lakes dataset"""
    global _LAKE_GEOMETRY
    
    if _LAKE_GEOMETRY is not None:
        return _LAKE_GEOMETRY
    
    try:
        # Get lakes shapefile from Natural Earth
        shpfilename = shpreader.natural_earth(resolution='10m',
                                               category='physical',
                                               name='lakes')
        reader = shpreader.Reader(shpfilename)
        
        # Find Lake Erie
        lake_erie = None
        for record in reader.records():
            # Check if this lake intersects with our bbox
            geom = record.geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            if (bounds[0] < BBOX[2] and bounds[2] > BBOX[0] and 
                bounds[1] < BBOX[3] and bounds[3] > BBOX[1]):
                # This is likely Lake Erie - it overlaps our bbox
                lake_erie = geom
                break
        
        if lake_erie is None:
            print("Warning: Could not find Lake Erie geometry, using bbox instead")
            from shapely.geometry import box
            lake_erie = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])
        
        _LAKE_GEOMETRY = prep(lake_erie)
        
    except Exception as e:
        print(f"Warning: Could not load lake geometry ({e}), using bbox instead")
        from shapely.geometry import box
        lake_erie = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])
        _LAKE_GEOMETRY = prep(lake_erie)
    
    return _LAKE_GEOMETRY

def mask_predictions_to_lake(predictions, lons, lats):
    """Mask predictions to only show pixels within Lake Erie boundary"""
    lake_geom = get_lake_erie_geometry()
    
    # Create 2D coordinate grids
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create mask for pixels within the lake
    lake_mask = np.zeros_like(predictions, dtype=bool)
    
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            if lake_geom.contains(point):
                lake_mask[i, j] = True
    
    # Apply lake mask AND existing NaN mask
    masked_predictions = np.where(lake_mask & ~np.isnan(predictions), 
                                   predictions, 
                                   np.nan)
    
    return masked_predictions

def load_mc_map(date_str):
    """Load MC probability map and coordinates for a given date"""
    map_file = DATA_DIR / f"mc_probability_{date_str}.npy"
    coords_file = DATA_DIR / f"mc_probability_{date_str}_coords.npz"
    
    if not map_file.exists() or not coords_file.exists():
        print(f"Files not found for date {date_str}")
        return None, None, None
    
    # Load prediction map
    predictions = np.load(map_file)
    
    # Load coordinates
    coords = np.load(coords_file)
    lons = coords['lons']
    lats = coords['lats']
    
    return predictions, lons, lats

def plot_mc_map(date_str, ax=None, show_title=True):
    """Plot a single MC probability map"""
    predictions, lons, lats = load_mc_map(date_str)
    
    if predictions is None:
        return None
    
    # Mask predictions to only show pixels within Lake Erie boundary
    lake_masked_predictions = mask_predictions_to_lake(predictions, lons, lats)
    
    # Create masked array for plotting
    masked_predictions = np.ma.masked_invalid(lake_masked_predictions)
    
    # Count valid pixels (after lake masking)
    valid_pixels = np.sum(~np.isnan(lake_masked_predictions))
    mean_prob = np.nanmean(lake_masked_predictions)
    max_prob = np.nanmax(lake_masked_predictions)
    
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add geographic features FIRST (so they appear below the data)
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightblue', edgecolor='navy', linewidth=0.5, zorder=1)
    ax.coastlines(resolution='10m', linewidth=0.8, zorder=3)
    
    # Set map extent to Lake Erie
    ax.set_extent([BBOX[0], BBOX[2], BBOX[1], BBOX[3]], crs=ccrs.PlateCarree())
    
    # Plot MC probability map - ONLY show valid predictions (lake pixels with data)
    # Create a 2D meshgrid from 1D coordinate arrays
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    im = ax.pcolormesh(lon_grid, lat_grid, masked_predictions, 
                       transform=ccrs.PlateCarree(),
                       cmap='RdYlGn_r',  # Red = high risk, Green = low risk
                       vmin=0, vmax=1,
                       shading='auto',
                       alpha=0.8,  # Slight transparency to see lake underneath
                       zorder=2)  # Draw data layer between lake and borders
    
    # Set the color for masked/NaN values to be transparent
    im.cmap.set_bad(alpha=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label('Microcystin Probability', fontsize=11)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Format date
    dt = datetime.strptime(date_str, '%Y%m%d')
    date_formatted = dt.strftime('%B %d, %Y')
    
    # Add title with statistics
    if show_title:
        title = f'Microcystin Probability - Lake Erie\n{date_formatted}'
        subtitle = f'Valid pixels: {valid_pixels:,} | Mean prob: {mean_prob:.3f} | Max prob: {max_prob:.3f}'
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, pad=10)
    
    return ax

def plot_time_series(date_list, output_file='mc_time_series.png'):
    """Plot a time series of MC probability maps"""
    n_maps = len(date_list)
    
    # Determine grid layout
    if n_maps <= 4:
        rows, cols = 2, 2
    elif n_maps <= 6:
        rows, cols = 2, 3
    elif n_maps <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    fig = plt.figure(figsize=(6*cols, 5*rows))
    
    for idx, date_str in enumerate(date_list[:rows*cols]):
        ax = fig.add_subplot(rows, cols, idx+1, projection=ccrs.PlateCarree())
        plot_mc_map(date_str, ax=ax, show_title=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved time series visualization to {output_file}")
    plt.show()

def plot_bloom_season_comparison(output_file='bloom_comparison.png'):
    """Compare 2024 and 2025 bloom season predictions"""
    # Get available maps
    all_maps = sorted([f.stem.replace('mc_probability_', '') 
                      for f in DATA_DIR.glob('mc_probability_*.npy')])
    
    # Filter for June-August (peak bloom months)
    bloom_2024 = [d for d in all_maps if d.startswith('2024') and d[4:6] in ['06', '07', '08']]
    bloom_2025 = [d for d in all_maps if d.startswith('2025') and d[4:6] in ['06', '07', '08']]
    
    # Select representative dates
    sample_2024 = bloom_2024[:3] if len(bloom_2024) >= 3 else bloom_2024
    sample_2025 = bloom_2025[:3] if len(bloom_2025) >= 3 else bloom_2025
    
    if not sample_2024 and not sample_2025:
        print("No bloom season data available yet")
        return
    
    n_total = len(sample_2024) + len(sample_2025)
    fig = plt.figure(figsize=(18, 5*((n_total+2)//3)))
    
    plot_idx = 1
    
    # Plot 2024 samples
    for date_str in sample_2024:
        ax = fig.add_subplot(2, 3, plot_idx, projection=ccrs.PlateCarree())
        plot_mc_map(date_str, ax=ax, show_title=True)
        plot_idx += 1
    
    # Plot 2025 samples
    for date_str in sample_2025:
        ax = fig.add_subplot(2, 3, plot_idx, projection=ccrs.PlateCarree())
        plot_mc_map(date_str, ax=ax, show_title=True)
        plot_idx += 1
    
    plt.suptitle('Bloom Season Comparison: 2024 vs 2025', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved bloom comparison to {output_file}")
    plt.show()

def show_recent_maps(n=6, output_file='recent_mc_maps.png'):
    """Show the most recently generated maps"""
    # Get all map dates
    all_maps = sorted([f.stem.replace('mc_probability_', '') 
                      for f in DATA_DIR.glob('mc_probability_*.npy')])
    
    if not all_maps:
        print("No maps found!")
        return
    
    # Get most recent n maps
    recent_maps = all_maps[-n:]
    
    print(f"Visualizing {len(recent_maps)} most recent maps:")
    for date_str in recent_maps:
        dt = datetime.strptime(date_str, '%Y%m%d')
        print(f"  - {dt.strftime('%B %d, %Y')}")
    
    plot_time_series(recent_maps, output_file=output_file)

def calculate_statistics():
    """Calculate and display statistics about all generated maps"""
    all_maps = sorted([f.stem.replace('mc_probability_', '') 
                      for f in DATA_DIR.glob('mc_probability_*.npy')])
    
    if not all_maps:
        print("No maps found!")
        return
    
    print(f"\n{'='*60}")
    print(f"MC PROBABILITY MAPS STATISTICS")
    print(f"{'='*60}")
    print(f"Total maps generated: {len(all_maps)}")
    print(f"Date range: {all_maps[0]} to {all_maps[-1]}")
    
    # Monthly breakdown
    monthly_counts = {}
    for date_str in all_maps:
        month_key = date_str[:6]  # YYYYMM
        monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
    
    print(f"\nMonthly breakdown:")
    for month, count in sorted(monthly_counts.items()):
        dt = datetime.strptime(month, '%Y%m')
        print(f"  {dt.strftime('%B %Y')}: {count} maps")
    
    # Analyze a sample of maps for probability statistics
    sample_size = min(20, len(all_maps))
    sample_dates = all_maps[::len(all_maps)//sample_size][:sample_size]
    
    mean_probs = []
    max_probs = []
    valid_pixel_counts = []
    
    for date_str in sample_dates:
        predictions, _, _ = load_mc_map(date_str)
        if predictions is not None:
            mean_probs.append(np.nanmean(predictions))
            max_probs.append(np.nanmax(predictions))
            valid_pixel_counts.append(np.sum(~np.isnan(predictions)))
    
    print(f"\nProbability statistics (sample of {len(sample_dates)} maps):")
    print(f"  Mean probability: {np.mean(mean_probs):.3f} ± {np.std(mean_probs):.3f}")
    print(f"  Max probability: {np.mean(max_probs):.3f} ± {np.std(max_probs):.3f}")
    print(f"  Valid pixels per map: {np.mean(valid_pixel_counts):.0f} ± {np.std(valid_pixel_counts):.0f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize MC Probability Maps',
        epilog='''
Examples:
  # Show statistics only
  python visualize_mc_maps.py --stats
  
  # List available dates
  python visualize_mc_maps.py --list
  
  # Show 10 most recent maps
  python visualize_mc_maps.py --recent 10
  
  # Visualize specific dates
  python visualize_mc_maps.py --dates 20240611 20240715 20240820
  
  # Create bloom season comparison
  python visualize_mc_maps.py --bloom --output bloom_2024_vs_2025.png
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dates', nargs='+', help='Specific dates to visualize (YYYYMMDD format)')
    parser.add_argument('--recent', type=int, default=6, help='Number of recent maps to show (default: 6)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--list', action='store_true', help='List all available dates')
    parser.add_argument('--bloom', action='store_true', help='Create bloom season comparison')
    parser.add_argument('--output', default='mc_maps.png', help='Output filename (default: mc_maps.png)')
    
    args = parser.parse_args()
    
    print("MC Probability Map Visualization Tool")
    print("="*60)
    
    # Get all available maps
    all_maps = sorted([f.stem.replace('mc_probability_', '') 
                      for f in DATA_DIR.glob('mc_probability_*.npy')])
    
    if args.list:
        # List all available dates
        print(f"\nAvailable dates ({len(all_maps)} total):\n")
        for date_str in all_maps:
            dt = datetime.strptime(date_str, '%Y%m%d')
            print(f"  {date_str} - {dt.strftime('%B %d, %Y (%A)')}")
        exit(0)
    
    # Always show statistics
    calculate_statistics()
    
    if args.stats:
        # Just show statistics and exit
        exit(0)
    
    print("\nGenerating visualizations...")
    
    if args.dates:
        # User specified specific dates
        print(f"Visualizing {len(args.dates)} specified dates:")
        for date_str in args.dates:
            try:
                dt = datetime.strptime(date_str, '%Y%m%d')
                print(f"  - {dt.strftime('%B %d, %Y')}")
            except:
                print(f"  - {date_str} (invalid format)")
        plot_time_series(args.dates, output_file=args.output)
    
    elif args.bloom:
        # Create bloom season comparison
        plot_bloom_season_comparison(output_file=args.output)
    
    else:
        # Default: show recent maps
        show_recent_maps(n=args.recent, output_file=args.output)
