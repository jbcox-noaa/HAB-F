"""
Chlorophyll-a LSTM Forecasting Module

This module provides a complete pipeline for spatiotemporal chlorophyll-a forecasting
using ConvLSTM2D neural networks. It supports both Sentinel-3 OLCI and PACE OCI sensors.

Main components:
- config: Configuration parameters and constants
- utils: Data preprocessing and utility functions
- model: ConvLSTM2D model architectures
- train: Training pipeline
- predict: Prediction and forecasting utilities

Example usage:
    # Training
    python -m chla_lstm_forecasting.train --data-dir data/composites --sensor S3
    
    # Prediction
    python -m chla_lstm_forecasting.predict --model best_model.keras --data-files file1.npy file2.npy
"""

__version__ = "1.0.0"
__author__ = "Jesse Cox"

# Import main components for public API
from . import config
from . import utils
from . import model
from . import train
from . import predict

# Expose key functions
from .utils import (
    parse_file,
    create_sequences,
    split_temporal_data,
    load_composite_data,
    plot_chlorophyll_map
)

from .model import (
    build_convlstm_model,
    build_deep_convlstm_model,
    build_model_from_config,
    load_model,
    save_model
)

from .train import train_model
from .predict import (
    predict_single_step,
    predict_multi_step,
    run_prediction,
    denormalize_chlorophyll
)

__all__ = [
    # Modules
    'config',
    'utils',
    'model',
    'train',
    'predict',
    
    # Utils
    'parse_file',
    'create_sequences',
    'split_temporal_data',
    'load_composite_data',
    'plot_chlorophyll_map',
    
    # Model
    'build_convlstm_model',
    'build_deep_convlstm_model',
    'build_model_from_config',
    'load_model',
    'save_model',
    
    # Train
    'train_model',
    
    # Predict
    'predict_single_step',
    'predict_multi_step',
    'run_prediction',
    'denormalize_chlorophyll',
]
