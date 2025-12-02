"""
Configuration for Spectral MC Forecasting (Phase 7)
"""

from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PACE_GRANULES_DIR = DATA_DIR  # PACE granule .nc files are in data/
MC_MAPS_DIR = DATA_DIR / "MC_probability_maps"
SPECTRAL_DATA_DIR = DATA_DIR / "PACE_spectral_sequences"  # New: processed sequences
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Create directories
SPECTRAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PACE Spectral Configuration
# ============================================================================

# PACE has 172 spectral bands from ~340nm to ~890nm
N_SPECTRAL_BANDS = 172
N_INPUT_FEATURES = 173  # 172 spectral bands + 1 mask channel

# Spatial grid (aligned with MC probability maps from Phase 2)
BBOX = (-83.5, 41.3, -82.45, 42.2)  # Lake Erie (lon_min, lat_min, lon_max, lat_max)
RES_KM = 1.2  # Spatial resolution
HEIGHT = 84  # From MC probability maps
WIDTH = 73   # From MC probability maps

# Patch-based approach (for GLERL-labeled samples)
PATCH_SIZE = 11  # Extract NxN patches around GLERL measurement points (must be odd)
# Patch size of 11 provides ~6.6km x 6.6km spatial context (11 * 1.2km/pixel = 13.2km)
# Center pixel corresponds to GLERL measurement location
USE_PATCH_BASED = True  # Use patch-based sequences for supervised training

# Wavelength ranges of interest for microcystin detection
# Based on phycocyanin (620nm), chlorophyll (665nm, 443nm), turbidity
WAVELENGTH_BANDS = {
    'phycocyanin': 620,      # Key indicator for cyanobacteria
    'chlorophyll_a_1': 443,  # Blue absorption peak
    'chlorophyll_a_2': 665,  # Red absorption peak
    'turbidity': 550,        # Green reflectance
    'nir': 750,              # Near-infrared (water absorption)
}

# ============================================================================
# Model Architecture
# ============================================================================

# Spectral Encoder (learns compressed representation)
ENCODER_LAYERS = [128, 64, 32, 16]  # 172 → 128 → 64 → 32 → 16
ENCODER_ACTIVATION = 'relu'
ENCODER_DROPOUT = 0.3

# ConvLSTM (temporal-spatial modeling)
CONVLSTM_FILTERS = [64, 64, 32]
CONVLSTM_KERNEL_SIZE = (3, 3)
CONVLSTM_DROPOUT = 0.3
CONVLSTM_RECURRENT_DROPOUT = 0.2

# Decoder (16 features → MC probability)
DECODER_LAYERS = [32, 16, 8]  # Final layer is 1 with sigmoid
DECODER_ACTIVATION = 'relu'
DECODER_DROPOUT = 0.2
DECODER_FINAL_ACTIVATION = 'sigmoid'  # Binary classification output

# ============================================================================
# Training Configuration
# ============================================================================

# Sequence parameters
SEQ_LEN = 14  # Input sequence length (14-day lookback window)
FORECAST_HORIZON = 1  # Predict 1 day ahead
MIN_VALID_DAYS = 1  # Minimum valid granules required (0 would predict last-known-state)
USE_MASK_CHANNEL = True  # Add mask channel to indicate valid/missing days
LAST_KNOWN_STATE_FALLBACK = True  # Use most recent valid map when all lookback is masked

# Training splits (temporal - no data leakage)
TRAIN_YEAR = 2024
VAL_END_DATE = "2025-07-31"
TEST_START_DATE = "2025-08-01"

# Phase 1: Unsupervised Pre-training (Autoencoder)
PHASE1_EPOCHS = 50
PHASE1_BATCH_SIZE = 32
PHASE1_LR = 1e-3
PHASE1_PATIENCE = 10

# Phase 2: Supervised Fine-tuning (GLERL measurements)
PHASE2_EPOCHS = 100
PHASE2_BATCH_SIZE = 16
PHASE2_LR = 5e-4
PHASE2_PATIENCE = 15

# Phase 3: Semi-supervised (Spatial expansion)
PHASE3_EPOCHS = 50
PHASE3_BATCH_SIZE = 16
PHASE3_LR = 1e-4
PHASE3_PATIENCE = 10

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.3
NOISE_LEVEL = 0.02  # Gaussian noise std for spectral bands

# Regularization
L2_REG = 1e-3
SPATIAL_DROPOUT = 0.2

# ============================================================================
# Data Processing
# ============================================================================

# Microcystin threshold for binary classification
MC_THRESHOLD = 1.0  # µg/L - WHO guideline for drinking water

# Output configuration
OUTPUT_TYPE = 'binary'  # 'binary' classification (MC ≥ threshold)
OUTPUT_ACTIVATION = 'sigmoid'  # Sigmoid for binary classification
OUTPUT_CHANNELS = 1  # Single probability output

# Normalization strategy
NORMALIZE_METHOD = 'z-score'  # 'z-score' or 'min-max'
CLIP_PERCENTILES = (1, 99)  # Clip outliers before normalization

# Gap filling (for missing pixels)
GAP_FILL_METHOD = 'hybrid'  # 'temporal', 'spatial', 'hybrid', or 'none'
TEMPORAL_MAX_DAYS = 3
SPATIAL_MAX_DIST = 3  # pixels

# Sentinel value for masked pixels
SENTINEL_VALUE = -999.0

# ============================================================================
# Evaluation
# ============================================================================

# Metrics
METRICS = ['mse', 'mae', 'rmse']

# Comparison baseline
PHASE2_BASELINE_MSE = 0.0247  # Current best from Phase 2

# Success criteria
MIN_IMPROVEMENT = 0.20  # 20% improvement required for production
TARGET_MSE = 0.020  # Target performance

# ============================================================================
# Hardware
# ============================================================================

# Memory management
PREFETCH_BUFFER = 2
CACHE_SIZE = 100  # Number of sequences to cache in memory

# GPU configuration
USE_MIXED_PRECISION = True
GPU_MEMORY_GROWTH = True

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TENSORBOARD_DIR = BASE_DIR / "tensorboard"
TENSORBOARD_DIR.mkdir(exist_ok=True)

# Checkpoint configuration
CHECKPOINT_FREQ = 5  # Save every N epochs
SAVE_BEST_ONLY = True

# ============================================================================
# Helper Functions
# ============================================================================

def get_normalization_stats_path(split='train'):
    """Get path for normalization statistics."""
    return SPECTRAL_DATA_DIR / f"normalization_stats_{split}.npz"

def get_spectral_sequences_path(split='train'):
    """Get path for processed spectral sequences."""
    return SPECTRAL_DATA_DIR / f"spectral_sequences_{split}.h5"

def get_model_path(phase, epoch=None):
    """Get path for model checkpoint."""
    if epoch is None:
        return MODEL_DIR / f"best_model_phase{phase}.keras"
    else:
        return MODEL_DIR / f"model_phase{phase}_epoch{epoch:03d}.keras"

def get_results_path(phase):
    """Get path for results JSON."""
    return OUTPUT_DIR / f"results_phase{phase}.json"

def get_config_summary():
    """Get configuration summary for logging."""
    return {
        'spectral_bands': N_SPECTRAL_BANDS,
        'spatial_grid': f"{HEIGHT}x{WIDTH}",
        'sequence_length': SEQ_LEN,
        'encoder_architecture': ' → '.join(map(str, [N_SPECTRAL_BANDS] + ENCODER_LAYERS)),
        'convlstm_filters': CONVLSTM_FILTERS,
        'decoder_architecture': ' → '.join(map(str, DECODER_LAYERS)),
        'phase1_config': {
            'epochs': PHASE1_EPOCHS,
            'batch_size': PHASE1_BATCH_SIZE,
            'learning_rate': PHASE1_LR,
        },
        'phase2_config': {
            'epochs': PHASE2_EPOCHS,
            'batch_size': PHASE2_BATCH_SIZE,
            'learning_rate': PHASE2_LR,
        },
        'target_mse': TARGET_MSE,
        'baseline_mse': PHASE2_BASELINE_MSE,
    }
