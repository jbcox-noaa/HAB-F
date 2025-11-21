# Chlorophyll-a LSTM Forecasting

Spatiotemporal chlorophyll-a forecasting using ConvLSTM2D neural networks for Lake Erie.

## Overview

This module provides a complete pipeline for forecasting chlorophyll-a concentrations using satellite time series data. It supports both Sentinel-3 OLCI (historical, 2016+) and PACE OCI (current, 2024+) sensors.

**Key Features:**
- ConvLSTM2D architecture for spatiotemporal modeling
- Multi-step autoregressive forecasting (up to 7 days)
- Mixed precision training (float16) for efficiency
- Temporal data splitting (prevents data leakage)
- Comprehensive visualization tools

## Architecture

**Model:** Sequential ConvLSTM2D  
- Input: Temporal sequence (5 timesteps, H×W, 2 channels)
  - Channel 1: Log-normalized chlorophyll-a
  - Channel 2: Valid pixel mask
- Layer 1: ConvLSTM2D(32 filters, 3×3 kernel, tanh)
- Layer 2: ConvLSTM2D(32 filters, 3×3 kernel, tanh)
- Output: Predicted chlorophyll-a map (H×W, 1 channel)

**Preprocessing:**
1. Extract chlorophyll band (band 21 for Sentinel-3)
2. Log transformation: `log10(chla + 1)`
3. Normalize to [-1, 1] range
4. Create binary mask for valid pixels

## Installation

Required dependencies are in the main project `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model on Sentinel-3 composites:

```bash
python -m chla_lstm_forecasting.train \
  --data-dir CNN-LSTM/Images2 \
  --sensor S3 \
  --seq-len 5 \
  --epochs 50 \
  --batch-size 4 \
  --output-dir ./output
```

**Arguments:**
- `--data-dir`: Directory containing `composite_data_S3_*.npy` files
- `--sensor`: Sensor type (`S3` or `PACE`)
- `--seq-len`: Sequence length (default: 5)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 4)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--model-type`: Architecture (`standard` or `deep`)
- `--output-dir`: Output directory for models and plots

**Outputs:**
- `best_model.keras`: Model checkpoint (best validation loss)
- `final_model.keras`: Final model after training
- `training_history.png`: Loss curves

### Prediction

Generate forecasts using a trained model:

```bash
python -m chla_lstm_forecasting.predict \
  --model best_model.keras \
  --data-files composite_data_S3_2024-07-01.npy composite_data_S3_2024-07-02.npy ... \
  --n-steps 7 \
  --output-dir ./forecasts
```

**Arguments:**
- `--model`: Path to trained model (`.keras` file)
- `--data-files`: Input files (chronologically sorted, ≥ seq_len)
- `--seq-len`: Sequence length (default: 5)
- `--n-steps`: Forecast horizon (default: 1, up to 7)
- `--output-dir`: Output directory for visualizations

**Outputs:**
- `forecast_Nstep.png`: Spatial maps of forecasted chlorophyll
- `timeseries_center.png`: Time series at center pixel

### Programmatic API

```python
from chla_lstm_forecasting import train_model, run_prediction

# Train model
train_model(
    data_dir='CNN-LSTM/Images2',
    sensor='S3',
    seq_len=5,
    epochs=50
)

# Generate forecast
run_prediction(
    model_path='best_model.keras',
    data_files=['file1.npy', 'file2.npy', ...],
    n_steps=7
)
```

## Data Format

**Sentinel-3 Composites:**
- File naming: `composite_data_S3_YYYY-MM-DD.npy`
- Format: NumPy array, shape (H, W, 21)
- Band 21: Chlorophyll-a concentration (mg/m³)
- Location: `CNN-LSTM/Images2/`

**PACE Composites:**
- File naming: `composite_data_PACE_YYYY-MM-DD.npy`
- Format: NumPy array, shape (H, W, 172)
- Product: `chl_ocx` (chlorophyll from multi-algorithm)

## Configuration

Key parameters in `config.py`:

```python
# Data
SEQUENCE_LENGTH = 5           # Timesteps in input sequence
PREDICTION_HORIZON = 1        # Timesteps ahead to predict

# Preprocessing
MAX_CHLA = 500               # Maximum chlorophyll for normalization
INVALID_PIXEL_VALUE = -1.0   # Value for masked pixels

# Model
CONVLSTM_FILTERS = [32, 32]  # Filters per ConvLSTM layer
CONVLSTM_KERNEL = (3, 3)     # Kernel size
ACTIVATION = 'tanh'          # Activation function

# Training
BATCH_SIZE = 4               # Batch size
LEARNING_RATE = 1e-4         # Adam learning rate
EPOCHS = 50                  # Training epochs
TRAIN_SPLIT = 0.6            # Training fraction
VAL_SPLIT = 0.2              # Validation fraction (test = 0.2)
```

## Module Structure

```
chla_lstm_forecasting/
├── __init__.py          # Public API
├── config.py            # Configuration parameters
├── utils.py             # Preprocessing and utilities
├── model.py             # ConvLSTM2D architectures
├── train.py             # Training pipeline
└── predict.py           # Prediction and forecasting
```

## Example Workflow

1. **Prepare data**: Ensure composite files are in `CNN-LSTM/Images2/`
2. **Train model**:
   ```bash
   python -m chla_lstm_forecasting.train --data-dir CNN-LSTM/Images2 --sensor S3
   ```
3. **Generate forecast**:
   ```bash
   python -m chla_lstm_forecasting.predict \
     --model best_model.keras \
     --data-files $(ls CNN-LSTM/Images2/composite_data_S3_2024-07-*.npy | tail -5) \
     --n-steps 7
   ```

## Technical Details

**Mixed Precision Training:**
- Policy: `mixed_float16` (float16 compute, float32 storage)
- Memory savings: ~50% reduction in GPU memory
- Speed: 1.5-2x faster on modern GPUs

**Temporal Splitting:**
- Training: First 60% chronologically
- Validation: Next 20%
- Test: Final 20%
- Prevents data leakage in time series

**Multi-Step Forecasting:**
- Autoregressive: Use previous prediction as input
- Propagates mask channel through sequence
- Up to 7-day horizon supported

## Validation

Run smoke tests:
```bash
python test_chla_module.py
```

Expected output: `Passed: 5/5 ✓ All tests passed!`

## Migration from Original LSTM.py

The original monolithic script (`CNN-LSTM/LSTM.py`) has been refactored into this modular structure:

| Original | New Module | Notes |
|----------|------------|-------|
| Global vars | `config.py` | Centralized configuration |
| `parse_file()` | `utils.parse_file()` | Same functionality |
| `create_sequences()` | `utils.create_sequences()` | Same functionality |
| `build_model()` | `model.build_convlstm_model()` | Same architecture |
| Main script | `train.train_model()` | CLI interface |
| N/A | `predict.run_prediction()` | New forecasting tools |

## Future Enhancements

- [ ] Multi-GPU training support
- [ ] Ensemble forecasting (multiple models)
- [ ] Uncertainty quantification
- [ ] Real-time PACE data ingestion
- [ ] Integration with Phase 2 (microcystin detection)

## References

- Sentinel-3 OLCI: https://sentinel.esa.int/web/sentinel/missions/sentinel-3
- PACE OCI: https://pace.gsfc.nasa.gov/
- ConvLSTM: Shi et al. (2015) "Convolutional LSTM Network"

## Author

Jesse Cox  
Version 1.0.0
