"""
Regenerate GLERL ground truth CSV with surface-only depth filtering.

This script:
1. Loads all GLERL source files (2012-2025)
2. Filters to Surface + Scum samples only (excludes Bottom, Mid-column)
3. Standardizes column names
4. Saves to glrl-hab-data-surface-only.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_2020_2021():
    """Load 2020-2021 data with depth filtering."""
    df = pd.read_csv('GLERL_GT/data/noaa-glerl-erie-habs-field-sampling-results-2020-2021.csv')
    
    # Filter to Surface + Scum only
    df = df[df['sample_depth_category'].isin(['Surface', 'Scum'])]
    
    # Combine date and time
    df['timestamp'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
    
    # Standardize columns (note: column names use underscores, not camelCase)
    df = df.rename(columns={
        'particulate_microcystin': 'particulate_microcystin',
        'extracted_chla': 'extracted_chla',
        'dissolved_microcystin': 'dissolved_microcystin'
    })
    
    return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]


def load_2022():
    """Load 2022 data with depth filtering."""
    df = pd.read_csv('GLERL_GT/data/noaa-glerl-erie-habs-field-sampling-results-2022.csv')
    
    # Filter to Surface + Scum only
    df = df[df['sample_depth_category'].isin(['Surface', 'Scum'])]
    
    # Combine date and time
    df['timestamp'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
    
    # Column names already match
    return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]


def load_2012_2018():
    """Load 2012-2018 data with depth filtering."""
    # Try different encodings
    for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv('GLERL_GT/data/lake_erie_habs_field_sampling_results_2012_2018.csv', 
                           encoding=encoding, low_memory=False)
            
            # Filter to Surface + Scum only
            depth_col = 'Sample Depth (category)'
            if depth_col in df.columns:
                df = df[df[depth_col].isin(['Surface', 'Scum'])]
            
            # Combine date and time
            df['timestamp'] = df['Date'].astype(str) + ' ' + df['Local Time (Eastern Time Zone)'].astype(str)
            
            # Standardize columns
            df = df.rename(columns={
                'Site': 'station_name',
                'Latitude (decimal deg)': 'lat',
                'Longitude (decimal deg)': 'lon',
                'Particulate Microcystin (µg/L)': 'particulate_microcystin',
                'Extracted Chlorophyll a (µg/L)': 'extracted_chla',
                'Dissolved Microcystin (µg/L)': 'dissolved_microcystin'
            })
            
            return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]
            
        except Exception as e:
            continue
    
    print(f"  Warning: Could not load 2012-2018 data")
    return None


def load_2019():
    """Load 2019 data with depth filtering."""
    for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv('GLERL_GT/data/lake_erie_habs_field_sampling_results_2019.csv', 
                           encoding=encoding, low_memory=False)
            
            # Filter to Surface + Scum only
            depth_col = 'Sample Depth (category)'
            if depth_col in df.columns:
                df = df[df[depth_col].isin(['Surface', 'Scum'])]
            
            # Combine date and time
            df['timestamp'] = df['Date'].astype(str) + ' ' + df['Local Time (Eastern Time Zone)'].astype(str)
            
            # Standardize columns
            df = df.rename(columns={
                'Site': 'station_name',
                'Latitude (decimal deg)': 'lat',
                'Longitude (decimal deg)': 'lon',
                'Particulate Microcystin (µg/L)': 'particulate_microcystin',
                'Extracted Chlorophyll a (µg/L)': 'extracted_chla',
                'Dissolved Microcystin (µg/L)': 'dissolved_microcystin'
            })
            
            return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]
            
        except Exception as e:
            continue
    
    print(f"  Warning: Could not load 2019 data")
    return None


def load_2024():
    """Load 2024 data with depth filtering."""
    try:
        df = pd.read_csv('GLERL_GT/data/2024_WLE_Weekly_Datashare_CSV.csv', low_memory=False)
        
        # Filter to Surface + Scum if depth column exists
        depth_col = [c for c in df.columns if 'depth' in c.lower() and 'category' in c.lower()]
        if depth_col:
            df = df[df[depth_col[0]].isin(['Surface', 'Scum'])]
        
        # Create timestamp from date and arrival time
        if 'Date' in df.columns and 'Arrival_Time' in df.columns:
            df['timestamp'] = df['Date'].astype(str) + ' ' + df['Arrival_Time'].astype(str)
        
        # Standardize columns
        df = df.rename(columns={
            'Site': 'station_name',
            'Lat_deg': 'lat',
            'Long_deg': 'lon',
            'Particulate_MC_ugL-1': 'particulate_microcystin',
            'Extracted_Chla_ugL-1': 'extracted_chla',
            'Dissolved_MC_ugL-1': 'dissolved_microcystin'
        })
        
        return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]
    except Exception as e:
        print(f"  Warning: Could not load 2024 data: {e}")
        return None


def load_2025():
    """Load 2025 data with depth filtering."""
    try:
        df = pd.read_csv('GLERL_GT/data/2025_WLE_Weekly_Datashare_CSV.csv', low_memory=False)
        
        # Filter to Surface + Scum if depth column exists
        depth_col = [c for c in df.columns if 'depth' in c.lower() and 'category' in c.lower()]
        if depth_col:
            df = df[df[depth_col[0]].isin(['Surface', 'Scum'])]
        
        # Create timestamp from date and arrival time
        if 'Date' in df.columns and 'Arrival_Time' in df.columns:
            df['timestamp'] = df['Date'].astype(str) + ' ' + df['Arrival_Time'].astype(str)
        
        # Standardize columns
        df = df.rename(columns={
            'Site': 'station_name',
            'Lat_deg': 'lat',
            'Long_deg': 'lon',
            'Particulate_MC_ugL-1': 'particulate_microcystin',
            'Extracted_Chla_ugL-1': 'extracted_chla',
            'Dissolved_MC_ugL-1': 'dissolved_microcystin'
        })
        
        return df[['timestamp', 'station_name', 'lat', 'lon', 'particulate_microcystin', 'extracted_chla', 'dissolved_microcystin']]
    except Exception as e:
        print(f"  Warning: Could not load 2025 data: {e}")
        return None


def main():
    print("="*70)
    print("REGENERATING GLERL DATA WITH SURFACE-ONLY DEPTH FILTERING")
    print("="*70)
    
    dfs = []
    
    # Load each year
    print("\nLoading data by year:")
    
    df_2012_2018 = load_2012_2018()
    if df_2012_2018 is not None:
        print(f"  2012-2018: {len(df_2012_2018)} surface/scum samples")
        dfs.append(df_2012_2018)
    
    df_2019 = load_2019()
    if df_2019 is not None:
        print(f"  2019: {len(df_2019)} surface/scum samples")
        dfs.append(df_2019)
    
    df_2020_2021 = load_2020_2021()
    print(f"  2020-2021: {len(df_2020_2021)} surface/scum samples")
    dfs.append(df_2020_2021)
    
    df_2022 = load_2022()
    print(f"  2022: {len(df_2022)} surface/scum samples")
    dfs.append(df_2022)
    
    df_2024 = load_2024()
    if df_2024 is not None:
        print(f"  2024: {len(df_2024)} surface/scum samples")
        dfs.append(df_2024)
    
    df_2025 = load_2025()
    if df_2025 is not None:
        print(f"  2025: {len(df_2025)} surface/scum samples")
        dfs.append(df_2025)
    
    # Combine all
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"COMBINED: {len(combined)} total surface/scum samples")
    
    # Convert timestamp
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps or coordinates
    before_clean = len(combined)
    combined = combined.dropna(subset=['timestamp', 'lat', 'lon'])
    print(f"After removing invalid timestamps/coords: {len(combined)} samples ({before_clean - len(combined)} dropped)")
    
    # Sort by date
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nDate range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    # Check MC distribution - convert to numeric first
    combined['particulate_microcystin'] = pd.to_numeric(combined['particulate_microcystin'], errors='coerce')
    combined['mc'] = combined['particulate_microcystin'].fillna(0)
    print(f"\nMC distribution:")
    print(f"  MC >= 1.0: {(combined['mc'] >= 1.0).sum()} ({100*(combined['mc'] >= 1.0).sum()/len(combined):.1f}%)")
    print(f"  MC < 1.0: {(combined['mc'] < 1.0).sum()} ({100*(combined['mc'] < 1.0).sum()/len(combined):.1f}%)")
    
    # Save
    output_path = 'GLERL_GT/glrl-hab-data-surface-only.csv'
    combined.to_csv(output_path, index=True)
    print(f"\n✅ Saved to {output_path}")
    
    # Compare to old file
    try:
        old = pd.read_csv('GLERL_GT/glrl-hab-data.csv')
        print(f"\n📊 Comparison to old file:")
        print(f"  Old (all depths): {len(old)} samples")
        print(f"  New (surface only): {len(combined)} samples")
        print(f"  Reduction: {len(old) - len(combined)} samples ({100*(len(old) - len(combined))/len(old):.1f}%)")
    except:
        pass


if __name__ == '__main__':
    main()
