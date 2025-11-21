"""
Configuration for microcystin detection model.

This module contains all hyperparameters, paths, and settings for the
microcystin detection CNN.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directory for microcystin detection module
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR.parent / "data" / "cache"

# In-situ measurements (ground truth)
GLERL_CSV = BASE_DIR / "glrl-hab-data.csv"
USER_LABELS_CSV = BASE_DIR / "user-labels.csv"

# Output paths
TRAINING_DATA_PATH = DATA_DIR / "training_data_{sensor}.npy"
BALANCED_DATA_PATH = DATA_DIR / "training_data_balanced_{sensor}.npy"
PROCESSED_GRANULES_PATH = DATA_DIR / "processed_granules_{sensor}.txt"
CORRUPTED_GRANULES_PATH = BASE_DIR / "corrupted_granules.txt"

# Normalization statistics
CONTEXT_MEANS_PATH = DATA_DIR / "context_means.npy"
CONTEXT_STDS_PATH = DATA_DIR / "context_stds.npy"
CHANNEL_STATS_DIR = DATA_DIR / "channel_stats"

# Model checkpoint path
MODEL_PATH = MODELS_DIR / "model.keras"

# Station color mapping for visualization
STATION_COLORS_JSON = DATA_DIR / "station_colors_{sensor}.json"

# ============================================================================
# SATELLITE DATA
# ============================================================================

# Sensor configuration
SENSOR = "PACE"  # Options: "PACE", "SENTINEL", "MODIS"

# NASA Earthdata short names for PACE data
PACE_SHORT_NAMES = ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"]

# Sensor-specific parameters
SENSOR_PARAMS = {
    "PACE": {
        "short_names": PACE_SHORT_NAMES,
        "start_date": "2024-04-16",  # PACE mission start
        "res_km": 1.2,  # Spatial resolution in kilometers
        "channels": list(range(172)),  # Channel indices (0-171)
        "bbox": (-83.5, 41.3, -82.45, 42.2),  # Lake Erie
    },
    "SENTINEL-3": {
        "short_names": ["OLCIS3A_L2_EFR_OC"],
        "start_date": "2016-04-25",
        "res_km": 0.3,
        "channels": list(range(21)),  # Channel indices (0-20)
        "bbox": (-83.5, 41.3, -82.45, 42.2),
    }
}

# Geographic bounding box for Lake Erie (lon_min, lat_min, lon_max, lat_max)
BBOX = (-83.5, 41.3, -82.45, 42.2)

# ============================================================================
# TEMPORAL SPLITTING (to prevent data leakage)
# ============================================================================

# Temporal split strategy: stratified by date to preserve seasonal patterns
# Based on GLERL measurement dates from PACE era (Apr 2024 - May 2025)
# See docs/TEMPORAL_SPLITTING_STRATEGY.md for full analysis

TEMPORAL_SPLIT = {
    "train": [
        "2024-04-17", "2024-04-25", "2024-05-01", "2024-05-08", "2024-05-15",
        "2024-05-22", "2024-06-05", "2024-06-12", "2024-06-19"
    ],
    "val": [
        "2024-06-26", "2024-07-10", "2024-07-24", "2024-08-07", "2024-08-21"
    ],
    "test": [
        "2024-09-04", "2024-09-18", "2024-10-02", "2024-10-16", "2024-10-30",
        "2024-11-13", "2024-11-27", "2024-12-11", "2024-12-25", "2025-01-08"
    ]
}

# Total dates: 24 (9 train / 5 val / 10 test)
# Split ratio: 37.5% train / 20.8% val / 41.7% test
# Note: Test set is larger to ensure robust evaluation across seasons

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Patch sizes to experiment with (in pixels)
PATCH_SIZES = [3, 5, 7, 9]

# Particulate microcystin thresholds (µg/L)
# Used for binary classification (above/below threshold)
PM_THRESHOLDS = [0.1, 1.0]  # Low sensitivity (0.1) and moderate sensitivity (1.0)

# Time window for matching satellite data to in-situ measurements
HALF_TIME_WINDOW_DAYS = 1  # ±1 day = 3 day window total

# Minimum fraction of valid pixels in a patch
MIN_VALID_FRACTION = 0.5

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 300
VALIDATION_SPLIT = 0.125  # 1/8 for validation
TEST_SPLIT = 0.143  # 1/7 of remaining for test (after val split)

# Early stopping / checkpointing
MONITOR_METRIC = "val_accuracy"
SAVE_BEST_ONLY = True

# Data augmentation
AUGMENT_FLIPS = True  # Flip patches horizontally, vertically, both

# Class balancing
BALANCE_METHOD = "oversample"  # Options: "oversample", "granule", "none"
OVERSAMPLE_NEG_SAMPLES = 500  # Number of negative samples to add
OVERSAMPLE_START_DATE = "2024-11-15"
OVERSAMPLE_END_DATE = "2025-04-01"

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Minimum valid pixel fraction for predictions
PRED_MIN_VALID_FRAC = 0.5

# Number of days to look back for data aggregation
DAYS_LOOKBACK = 7

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot parameters
DPI = 100
INCLUDE_CHLA_LAYER = False  # Overlay chlorophyll-a on maps

# RGB band indices for PACE (for visualization)
# These correspond to red, green, blue wavelengths
RGB_INDICES_PACE = (105, 75, 48)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_channels_for_sensor(sensor: str) -> int:
    """Get number of spectral channels for a sensor."""
    channels = SENSOR_PARAMS.get(sensor.upper(), {}).get("channels", list(range(172)))
    return len(channels)


def get_training_data_path(sensor: str) -> Path:
    """Get path to training data for a specific sensor."""
    return Path(str(TRAINING_DATA_PATH).format(sensor=sensor.upper()))


def get_balanced_data_path(sensor: str) -> Path:
    """Get path to balanced training data for a specific sensor."""
    return Path(str(BALANCED_DATA_PATH).format(sensor=sensor.upper()))


def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHANNEL_STATS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print(f"Microcystin Detection Configuration")
    print(f"=" * 50)
    print(f"Sensor: {SENSOR}")
    print(f"Channels: {get_channels_for_sensor(SENSOR)}")
    print(f"Patch sizes: {PATCH_SIZES}")
    print(f"PM thresholds: {PM_THRESHOLDS}")
    print(f"BBox: {BBOX}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
