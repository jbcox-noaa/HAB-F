"""
Configuration for MC Forecasting ConvLSTM
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "MC_probability_maps"
MODEL_DIR = PROJECT_ROOT / "mc_lstm_forecasting"
MODEL_PATH = MODEL_DIR / "best_model.keras"

# ============================================================================
# DATA PARAMETERS
# ============================================================================
# Lake Erie bounding box
BBOX = (-83.5, 41.3, -82.45, 42.2)  # (lon_min, lat_min, lon_max, lat_max)

# Spatial dimensions (from PACE regridding at 1.2 km resolution)
HEIGHT = 84  # Latitude dimension
WIDTH = 73   # Longitude dimension

# Temporal parameters
SEQUENCE_LENGTH = 5  # Number of days in lookback window
FORECAST_HORIZON = 1  # Number of days ahead to predict

# Date range for data
TRAIN_YEAR = "2024"  # All 2024 data for training (242 maps)
VAL_YEAR = "2025"    # First part of 2025 for validation (includes early bloom)
TEST_YEAR = "2025"   # Second part of 2025 for testing (includes peak bloom)

# Temporal split strategy for 2025 data
# Split at August 1, 2025 to ensure both val and test contain bloom season
VAL_END_DATE = "20250801"  # Validation: Jan-Jul 2025 (includes bloom onset)
TEST_START_DATE = "20250801"  # Test: Aug-Oct 2025 (includes peak bloom)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Input shape: (sequence_length, height, width, channels)
INPUT_CHANNELS = 1  # MC probability only (vs 2 for chlorophyll: chla + mask)
INPUT_SHAPE = (SEQUENCE_LENGTH, HEIGHT, WIDTH, INPUT_CHANNELS)

# ConvLSTM parameters (proven architecture from chlorophyll forecasting)
FILTERS_1 = 32  # First ConvLSTM layer
FILTERS_2 = 32  # Second ConvLSTM layer
KERNEL_SIZE = (3, 3)  # Spatial convolution kernel
PADDING = 'same'

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE = 1e-5  # Adam optimizer learning rate
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10  # Early stopping patience

# Temporal split strategy:
# - Train: All 2024 data (242 maps across all months)
# - Validation: 2025 Jan-Jul (includes bloom onset, ~45 maps)
# - Test: 2025 Aug-Oct (includes peak bloom, ~30 maps)
# This ensures both validation and test sets contain bloom season data

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42  # For reproducibility

# ============================================================================
# DATA NORMALIZATION
# ============================================================================
# MC probabilities are already in [0, 1] range from ensemble predictions
# No additional normalization needed
MC_MIN = 0.0
MC_MAX = 1.0

# ============================================================================
# VALIDATION
# ============================================================================
# Metrics to track
METRICS = ['mae']  # Mean Absolute Error

# Model checkpoint
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_MODE = 'min'
CHECKPOINT_SAVE_BEST_ONLY = True
