"""
Configuration for combined forecasting pipeline.

This module contains configuration for the end-to-end system that combines
chlorophyll-a forecasting with microcystin detection.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directory for combined forecasting module
BASE_DIR = Path(__file__).parent

# Reference to other modules
PROJECT_ROOT = BASE_DIR.parent
CHLA_MODULE = PROJECT_ROOT / "chla_lstm_forecasting"
MICROCYSTIN_MODULE = PROJECT_ROOT / "microcystin_detection"

# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Output paths
FORECASTS_DIR = BASE_DIR / "forecasts"
PLOTS_DIR = BASE_DIR / "plots"

# Combined model checkpoint
COMBINED_MODEL_PATH = MODELS_DIR / "combined_model.keras"

# ============================================================================
# ARCHITECTURE SELECTION
# ============================================================================

# Which architecture to use
# "option1": Spectral → Future Spectral → Microcystin
# "option2": Spectral → Microcystin Maps → Future Microcystin  
# "both": Train and compare both architectures
ARCHITECTURE = "option2"  # Start with option2 as primary

# ============================================================================
# OPTION 1: SPECTRAL → FUTURE SPECTRAL → MICROCYSTIN
# ============================================================================

OPTION1_CONFIG = {
    "name": "spectral_forecast_first",
    "description": "CNN-LSTM predicts future spectral, then CNN detects toxin",
    
    # CNN-LSTM stage
    "chla_model": {
        "predict_full_spectrum": True,  # Predict all 172 bands
        "sequence_length": 5,
        "forecast_days": 7,
    },
    
    # CNN classification stage
    "microcystin_model": {
        "patch_size": 3,
        "threshold": 0.1,  # µg/L
    },
    
    # Training
    "train_end_to_end": True,  # Train both models together
    "freeze_stages": False,  # Allow gradients through both
}

# ============================================================================
# OPTION 2: SPECTRAL → MICROCYSTIN → FUTURE MICROCYSTIN
# ============================================================================

OPTION2_CONFIG = {
    "name": "toxin_forecast",
    "description": "CNN generates toxin maps, ConvLSTM predicts future",
    
    # CNN classification stage (first)
    "microcystin_model": {
        "patch_size": 3,
        "threshold": 0.1,  # µg/L
        "output_mode": "probability_map",  # Full spatial map
    },
    
    # ConvLSTM forecasting stage (second)
    "forecast_model": {
        "sequence_length": 5,
        "forecast_days": 7,
        "input_channels": 2,  # Probability + mask channel
        "convlstm_filters": [16, 16],
        "kernel_size": (3, 3),
    },
    
    # Hybrid option: include spectral features
    "include_spectral_features": True,  # Pass CNN features to LSTM
    "feature_channels": 32,  # Dimensionality of CNN features
    
    # Training
    "train_end_to_end": True,
    "freeze_cnn_epochs": 10,  # Freeze CNN for first N epochs
}

# ============================================================================
# ACTIVE CONFIGURATION
# ============================================================================

def get_active_config():
    """Get the configuration for the selected architecture."""
    if ARCHITECTURE == "option1":
        return OPTION1_CONFIG
    elif ARCHITECTURE == "option2":
        return OPTION2_CONFIG
    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

# ============================================================================
# DATA PIPELINE
# ============================================================================

# Satellite data source
DATA_SOURCE = "PACE"  # Must match both sub-models

# Geographic bounding box (must match sub-models)
BBOX = (-83.5, 41.3, -82.45, 42.2)

# Temporal parameters
SEQUENCE_LENGTH = 5  # Days of historical data
FORECAST_HORIZON = 7  # Days to forecast ahead

# Data split
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
BATCH_SIZE = 2  # Smaller due to combined model size
LEARNING_RATE = 1e-4
EPOCHS = 100

# Loss weighting (for multi-task learning)
LOSS_WEIGHTS = {
    "spectral": 1.0,      # Weight for spectral prediction loss (option 1)
    "microcystin": 2.0,   # Weight for microcystin prediction loss
    "temporal": 1.0,      # Weight for temporal consistency
}

# Optimizer
OPTIMIZER = "adam"

# Learning rate schedule
USE_LR_SCHEDULE = True
LR_SCHEDULE_FACTOR = 0.5
LR_SCHEDULE_PATIENCE = 5
LR_SCHEDULE_MIN = 1e-6

# ============================================================================
# MODEL ARCHITECTURE DETAILS
# ============================================================================

# For Option 2 - Hybrid architecture
HYBRID_CONFIG = {
    # Extract features from CNN before final classification
    "extract_features_at_layer": "conv2d_2",  # Layer name
    "feature_aggregation": "global_avg_pool",  # How to pool features
    
    # Combine with LSTM
    "feature_fusion": "concatenate",  # Options: "concatenate", "add", "attention"
    
    # Feature processing
    "feature_dense_units": [64, 32],
    "feature_dropout": 0.1,
}

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Forecasting mode
AUTOREGRESSIVE = True  # Use predictions as input for next step

# Uncertainty estimation
ESTIMATE_UNCERTAINTY = True
UNCERTAINTY_METHOD = "monte_carlo_dropout"  # MC Dropout during inference
MC_DROPOUT_SAMPLES = 20

# Post-processing
APPLY_TEMPORAL_SMOOTHING = True
SMOOTHING_WINDOW = 3  # Days

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS = {
    "spatial": [
        "accuracy",
        "precision", 
        "recall",
        "f1_score",
        "auc_roc",
    ],
    "temporal": [
        "mse",
        "mae",
        "rmse",
    ],
    "forecast_horizons": [1, 3, 5, 7],  # Evaluate at these horizons
}

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot parameters
PLOT_DPI = 150
PLOT_FORMAT = "png"

# Animation
CREATE_ANIMATIONS = True
ANIMATION_FPS = 2

# Comparison plots
COMPARE_ARCHITECTURES = True  # If both are trained

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Experiment name
EXPERIMENT_NAME = f"{ARCHITECTURE}_{DATA_SOURCE}"

# Logging
VERBOSE_TRAINING = True
LOG_INTERVAL = 10  # Log every N batches

# Checkpointing
SAVE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 5  # Save every N epochs

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_name():
    """Generate descriptive model name."""
    config = get_active_config()
    return f"{config['name']}_{DATA_SOURCE}_seq{SEQUENCE_LENGTH}_horizon{FORECAST_HORIZON}"


def validate_config():
    """Validate that configuration is consistent."""
    errors = []
    
    # Check that data source matches
    try:
        from chla_lstm_forecasting.config import DATA_SOURCE as CHLA_SOURCE
        from microcystin_detection.config import SENSOR as MC_SENSOR
        
        if DATA_SOURCE != CHLA_SOURCE:
            errors.append(f"Data source mismatch: combined={DATA_SOURCE}, chla={CHLA_SOURCE}")
        if DATA_SOURCE != MC_SENSOR:
            errors.append(f"Data source mismatch: combined={DATA_SOURCE}, microcystin={MC_SENSOR}")
    except ImportError as e:
        errors.append(f"Could not import sub-module configs: {e}")
    
    # Check architecture selection
    if ARCHITECTURE not in ["option1", "option2", "both"]:
        errors.append(f"Invalid architecture: {ARCHITECTURE}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    
    print(f"Combined Forecasting Pipeline Configuration")
    print(f"=" * 50)
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Data source: {DATA_SOURCE}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Forecast horizon: {FORECAST_HORIZON}")
    print(f"Model name: {get_model_name()}")
    print(f"\nActive config:")
    
    config = get_active_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nValidating configuration...")
    try:
        validate_config()
        print("✓ Configuration valid")
    except ValueError as e:
        print(f"✗ Configuration invalid:\n{e}")
