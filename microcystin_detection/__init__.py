"""
Microcystin Detection Module

This module provides tools for detecting microcystin concentrations in Lake Erie
using CNN models trained on PACE/Sentinel-3 satellite imagery and GLERL ground truth data.

Main Components:
- config: Configuration parameters and paths
- utils: Data processing utilities (regridding, patch extraction, visualization)
- model: CNN architecture (dual-input: patch + context)
- train: Training pipeline with data augmentation and callbacks
- data_collection: Satellite data collection with temporal splitting
- balance_training_data: Class balancing utilities
- predict: Prediction and inference tools

Example Usage:
    # Train a model
    from microcystin_detection.train import train_model
    train_model(sensor='PACE', patch_size=5)
    
    # Make predictions
    from microcystin_detection.predict import predict_from_granule
    predictions, lats, lons = predict_from_granule(granule_path, model, stats, ...)
    
    # Collect training data
    from microcystin_detection.data_collection import collect_training_data
    collect_training_data(sensor='PACE', temporal_split='train')
"""

__version__ = "2.0.0"
__author__ = "Jesse Cox"

from . import config
from . import utils
from . import model
from . import train
from . import data_collection
from . import balance_training_data
from . import predict

# Main functions
from .train import train_model
from .data_collection import collect_training_data
from .balance_training_data import balance_by_oversampling_negatives, analyze_class_distribution
from .predict import predict_from_granule, ensemble_predict, predict_time_series
from .model import build_model, load_model_with_normalization

__all__ = [
    'config',
    'utils',
    'model',
    'train',
    'data_collection',
    'balance_training_data',
    'predict',
    'train_model',
    'collect_training_data',
    'balance_by_oversampling_negatives',
    'analyze_class_distribution',
    'predict_from_granule',
    'ensemble_predict',
    'predict_time_series',
    'build_model',
    'load_model_with_normalization',
]

