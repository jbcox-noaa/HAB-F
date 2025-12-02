"""
Spectral MC Forecasting Module (Phase 7)

Direct spectral-to-toxin forecasting using raw PACE spectra.
Eliminates the CNN detector stage and works directly with 172-band spectral data.

Architecture:
    SpectralEncoder (172 bands → 16 features) + ConvLSTM + Decoder

Training Strategy (3 phases):
    1. Unsupervised pre-training (autoencoder on 1.8M pixels)
    2. Supervised fine-tuning (GLERL measurements)
    3. Semi-supervised spatial expansion (~900k pixels)

Expected Performance:
    MSE = 0.015-0.020 (20-40% better than Phase 2's 0.0247)
"""

__version__ = "0.1.0"
__author__ = "Jesse Cox"

# Import only config for now to avoid circular imports
from spectral_mc_forecasting import config

__all__ = [
    'config',
]
