"""
Chlorophyll-a LSTM Forecasting Module

This module provides tools for forecasting chlorophyll-a concentrations
in Lake Erie using CNN-LSTM models on satellite time series data.

Main components:
- data_preparation: Download and prepare time series satellite data
- model: CNN-LSTM architecture for temporal forecasting
- train: Training pipeline
- predict: Generate chlorophyll-a forecasts
- config: Configuration parameters
"""

__version__ = "1.0.0"
__author__ = "Jesse Cox"

from . import config

__all__ = ["config"]
