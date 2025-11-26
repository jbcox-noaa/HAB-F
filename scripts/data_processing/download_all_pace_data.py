#!/usr/bin/env python3
"""
Download all available PACE granules for Lake Erie region.

This script downloads all PACE L2 AOP granules that intersect with
Lake Erie bounding box from launch through current date.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BBOX = (-83.5, 41.3, -82.45, 42.2)  # Lake Erie (lon_min, lat_min, lon_max, lat_max)
OUTPUT_DIR = "./data"
PACE_START = "2024-02-08"  # PACE launch date
END_DATE = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")  # 5 days ago
SHORT_NAMES = ["PACE_OCI_L2_AOP"]  # Standard product (not NRT)


def download_pace_granules():
    """Download all PACE granules for Lake Erie."""
    
    # Authenticate
    logger.info("Authenticating with NASA Earthdata...")
    auth = earthaccess.login()
    
    if not auth.authenticated:
        logger.error("Authentication failed!")
        return
    
    logger.info(f"Searching for PACE granules:")
    logger.info(f"  Date range: {PACE_START} to {END_DATE}")
    logger.info(f"  Bounding box: {BBOX}")
    logger.info(f"  Short name: {SHORT_NAMES}")
    
    # Search for granules
    try:
        results = earthaccess.search_data(
            short_name=SHORT_NAMES,
            bounding_box=BBOX,
            temporal=(PACE_START, END_DATE),
            count=-1  # Get all results
        )
        
        logger.info(f"Found {len(results)} granules")
        
        if len(results) == 0:
            logger.warning("No granules found!")
            return
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Check which files already exist
        existing_files = set(os.listdir(OUTPUT_DIR))
        new_granules = []
        
        for granule in results:
            # Get filename
            granule_name = granule["umm"]["GranuleUR"]
            if ".nc" not in granule_name:
                granule_name = granule_name + ".nc"
            
            if granule_name not in existing_files:
                new_granules.append(granule)
        
        logger.info(f"Already have {len(results) - len(new_granules)} granules")
        logger.info(f"Downloading {len(new_granules)} new granules...")
        
        if len(new_granules) == 0:
            logger.info("No new data to download - you're up to date!")
            return
        
        # Download new granules
        paths = earthaccess.download(new_granules, OUTPUT_DIR)
        
        logger.info(f"✅ Successfully downloaded {len(paths)} granules")
        logger.info(f"Total PACE granules in {OUTPUT_DIR}: {len(results)}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PACE DATA DOWNLOAD FOR LAKE ERIE")
    logger.info("=" * 60)
    
    download_pace_granules()
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
