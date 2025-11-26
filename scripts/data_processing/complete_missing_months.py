#!/usr/bin/env python3
"""
Complete Missing Months - Process only months that don't have MC probability maps yet
Avoids re-downloading and re-processing existing data
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add the project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_process_pace_data import (
    download_month, process_month_granules, cleanup_temp_files,
    DATA_DIR, OUTPUT_DIR, TEMP_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_missing_months.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_existing_months():
    """Get list of months that already have MC probability maps"""
    existing_months = set()
    
    for map_file in OUTPUT_DIR.glob('mc_probability_*.npy'):
        date_str = map_file.stem.replace('mc_probability_', '')
        month_key = date_str[:6]  # YYYYMM
        existing_months.add(month_key)
    
    return existing_months

def get_all_month_ranges():
    """Get all month ranges from Feb 2024 to current date - 5 days"""
    start = datetime(2024, 2, 1).date()
    end = (datetime.now() - timedelta(days=5)).date()
    
    months = []
    current = start
    
    while current <= end:
        # Get first and last day of current month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        
        month_end = (next_month - timedelta(days=1))
        
        # Don't go past the end date
        if month_end > end:
            month_end = end
        
        month_str = current.strftime('%Y%m')
        month_start_str = current.strftime('%Y-%m-%d')
        month_end_str = month_end.strftime('%Y-%m-%d')
        
        months.append((month_str, month_start_str, month_end_str))
        
        current = next_month
    
    return months

def main():
    logger.info("="*80)
    logger.info("COMPLETE MISSING MONTHS - MC PROBABILITY MAP GENERATION")
    logger.info("="*80)
    
    # Get existing months
    existing_months = get_existing_months()
    logger.info(f"Found {len(existing_months)} months with existing data:")
    for month in sorted(existing_months):
        dt = datetime.strptime(month, '%Y%m')
        logger.info(f"  ✓ {dt.strftime('%B %Y')}")
    
    # Get all month ranges
    all_months = get_all_month_ranges()
    
    # Filter to only missing months
    missing_months = [(m, s, e) for m, s, e in all_months if m not in existing_months]
    
    if not missing_months:
        logger.info("\n✅ All months already processed! No work needed.")
        return
    
    logger.info(f"\nFound {len(missing_months)} missing months to process:")
    for month_key, start_date, end_date in missing_months:
        dt = datetime.strptime(month_key, '%Y%m')
        logger.info(f"  ⏳ {dt.strftime('%B %Y')} ({start_date} to {end_date})")
    
    logger.info(f"\n{'='*80}")
    logger.info("Starting processing of missing months...")
    logger.info(f"{'='*80}\n")
    
    total_maps = 0
    
    for idx, (month_key, start_date, end_date) in enumerate(missing_months, 1):
        dt = datetime.strptime(month_key, '%Y%m')
        logger.info("="*80)
        logger.info(f"MONTH {idx}/{len(missing_months)}: {dt.strftime('%B %Y')}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("="*80)
        
        try:
            # Download granules for this month
            logger.info(f"Downloading PACE data for {start_date} to {end_date}...")
            granule_paths = download_month(start_date, end_date)
            
            if not granule_paths:
                logger.warning(f"No granules found for {dt.strftime('%B %Y')}")
                continue
            
            logger.info(f"Downloaded {len(granule_paths)} granules")
            
            # Process granules
            maps_generated = process_month_granules(granule_paths, start_date)
            total_maps += maps_generated
            
            logger.info(f"Month complete: {maps_generated} maps generated")
            
            # Cleanup temp files
            cleanup_temp_files()
            logger.info(f"Cleaned up temporary files\n")
            
        except Exception as e:
            logger.error(f"Error processing {dt.strftime('%B %Y')}: {e}")
            logger.exception("Full traceback:")
            # Continue with next month
            continue
    
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Total new maps generated: {total_maps}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
