"""
Microcystin Detection Module

This module provides tools for detecting microcystin toxin in Lake Erie
using PACE hyperspectral satellite imagery and in-situ measurements.

Main components:
- data_collection: Download and process PACE granules
- granule_processing: Extract spatial patches around measurement stations
- balance_training_data: Handle class imbalance in training data
- model: CNN architecture for microcystin detection
- train: Training pipeline
- predict: Generate predictions for new satellite imagery
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Jesse Cox"

from . import config

__all__ = ["config"]
