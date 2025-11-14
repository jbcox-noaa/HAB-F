"""
Configuration for chlorophyll-a LSTM forecasting model.

This module contains all hyperparameters, paths, and settings for the
CNN-LSTM chlorophyll forecasting model.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directory for chla forecasting module
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR.parent / "data" / "cache"

# Output paths
COMPOSITE_DATA_PREFIX = "composite_data"
COMPOSITE_METADATA_PREFIX = "composite_metadata"

# Model checkpoint paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

# Plot output directory
PLOTS_DIR = BASE_DIR / "plots"

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# Primary data source (to experiment with both)
DATA_SOURCE = "PACE"  # Options: "PACE", "SENTINEL3"

# PACE configuration
PACE_CONFIG = {
    "short_names": ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"],
    "start_date": "2024-04-16",
    "n_channels": 172,
    "res_km": 1.2,
    # Chlorophyll-a can be derived from multiple PACE bands
    # or extracted from the chl_ocx product
    "chla_product": "chl_ocx",  # OCI chlorophyll algorithm
}

# Sentinel-3 configuration  
SENTINEL3_CONFIG = {
    "short_names": ["OLCIS3A_L2_EFR_OC"],
    "start_date": "2016-04-25",
    "n_channels": 21,
    "res_km": 0.3,
    "chla_band_index": 21,  # Chlorophyll-a is band 21 in S3 composites
}

# Geographic bounding box for Lake Erie (lon_min, lat_min, lon_max, lat_max)
BBOX = (-83.5, 41.3, -82.45, 42.2)

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Band index for Sentinel-3 chlorophyll-a
S3_CHLA_BAND_INDEX = 21

# Valid pixel threshold (minimum valid pixels per image)
VALID_PIXEL_THRESHOLD = 6500

# Maximum chlorophyll-a concentration for clipping outliers (mg/m³)
MAX_CHLA = 500.0

# Normalization method
NORMALIZATION = "log_minmax"  # Options: "log_minmax", "zscore", "minmax"

# Handle invalid pixels
INVALID_PIXEL_VALUE = -1.0  # Value to use for NaN/invalid pixels after normalization

# ============================================================================
# SEQUENCE CONFIGURATION
# ============================================================================

# Length of input sequences (number of time steps)
SEQUENCE_LENGTH = 5

# Prediction horizon (how many time steps ahead to predict)
PREDICTION_HORIZON = 1  # Currently predicts next single frame

# Stride for creating overlapping sequences
SEQUENCE_STRIDE = 1

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# ConvLSTM layers configuration
CONVLSTM_FILTERS = [32, 32]  # Number of filters in each ConvLSTM layer
CONVLSTM_KERNEL_SIZE = (3, 3)
CONVLSTM_ACTIVATION = "tanh"
CONVLSTM_PADDING = "same"

# Batch normalization
USE_BATCH_NORM = True

# Output convolution
OUTPUT_FILTERS = 1  # Single channel output (chlorophyll-a)
OUTPUT_KERNEL_SIZE = (3, 3)
OUTPUT_ACTIVATION = "tanh"  # Assuming normalized to [-1, 1]

# Mixed precision training (for memory efficiency)
USE_MIXED_PRECISION = True
MIXED_PRECISION_POLICY = "mixed_float16"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
BATCH_SIZE = 4  # Small batch size due to large spatial dimensions
LEARNING_RATE = 1e-4
EPOCHS = 50

# Train/val/test split ratios
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2  # 0.2 of total = validation
TEST_SPLIT = 0.2  # Remaining 0.2 = test

# Loss function
LOSS_FUNCTION = "mse"  # Mean squared error

# Optimizer
OPTIMIZER = "adam"

# ============================================================================
# CALLBACKS
# ============================================================================

# Model checkpoint
CHECKPOINT_MONITOR = "val_loss"
CHECKPOINT_MODE = "min"
CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_VERBOSE = 1

# Early stopping
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_RESTORE_BEST = True
EARLY_STOPPING_VERBOSE = 1

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Number of future time steps to predict
FORECAST_STEPS = 7  # Forecast 7 days ahead

# Autoregressive prediction (use previous predictions as input)
AUTOREGRESSIVE = True

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot parameters
PLOT_DPI = 150
PLOT_FORMAT = "png"

# Colormap for chlorophyll-a
CHLA_CMAP = "viridis"
CHLA_VMIN = -1.0  # For normalized values
CHLA_VMAX = 1.0

# Map projection
MAP_PROJECTION = "PlateCarree"  # Cartopy projection

# ============================================================================
# DATA QUALITY
# ============================================================================

# Minimum number of images needed for training
MIN_TRAINING_IMAGES = 10

# Maximum allowable gap in time series (days)
MAX_TIME_GAP_DAYS = 3

# Cloud cover handling
REJECT_HIGH_CLOUD_COVER = True
MAX_CLOUD_COVER_FRACTION = 0.7

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

def get_data_config(source: str = None) -> dict:
    """Get configuration for specified data source."""
    source = source or DATA_SOURCE
    if source.upper() == "PACE":
        return PACE_CONFIG
    elif source.upper() in ["SENTINEL3", "S3"]:
        return SENTINEL3_CONFIG
    else:
        raise ValueError(f"Unknown data source: {source}")


def get_n_channels(source: str = None) -> int:
    """Get number of channels for data source."""
    config = get_data_config(source)
    return config["n_channels"]


def get_input_shape(source: str = None) -> tuple:
    """
    Get expected input shape for the model.
    Returns (sequence_length, height, width, channels)
    Note: height and width are dynamic, determined at runtime
    """
    n_channels = get_n_channels(source)
    # Add 1 for mask channel
    return (SEQUENCE_LENGTH, None, None, n_channels + 1)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print(f"Chlorophyll-a LSTM Forecasting Configuration")
    print(f"=" * 50)
    print(f"Data source: {DATA_SOURCE}")
    print(f"Channels: {get_n_channels()}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Forecast steps: {FORECAST_STEPS}")
    print(f"BBox: {BBOX}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Input shape template: {get_input_shape()}")
