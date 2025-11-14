"""
Combined Forecasting Pipeline

This module combines chlorophyll-a forecasting with microcystin detection
to create an end-to-end forecasting system for HAB toxicity risk.

Two architectural approaches:
1. Option 1: Forecast spectral data, then detect toxin
2. Option 2: Detect toxin, then forecast toxin risk (primary)

Main components:
- pipeline: Sequential or end-to-end inference
- train_combined: Train combined model
- forecast: Multi-day forecasting
- config: Pipeline configuration
"""

__version__ = "1.0.0"
__author__ = "Jesse Cox"

from . import config

__all__ = ["config"]
