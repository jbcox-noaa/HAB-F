# HAB-F: Harmful Algal Bloom Forecasting

**Predictive modeling system for forecasting microcystin toxin risk in Lake Erie using satellite imagery and deep learning.**

## Overview

This project combines two deep learning models to forecast harmful algal bloom (HAB) toxicity:

1. **Chlorophyll-a Forecasting Model (CNN-LSTM)**: Time-series forecasting of chlorophyll-a concentration patterns
2. **Microcystin Detection Model (CNN)**: Spatial classification of microcystin toxin presence
3. **Combined Forecasting Pipeline**: End-to-end system predicting future toxin risk

### Two Architectural Approaches

**Option 1: Spectral → Future Spectral → Toxin**
- CNN-LSTM predicts future hyperspectral imagery (172 bands)
- CNN classifies predicted spectra to toxin risk maps
- Physics-based, interpretable intermediate outputs

**Option 2: Spectral → Toxin Maps → Future Toxin** (Primary)
- CNN classifies current spectra to toxin probability maps
- ConvLSTM forecasts future toxin risk patterns
- Direct prediction of target variable, more data-efficient

## Project Structure

```
HAB-F/
├── chla_lstm_forecasting/      # Chlorophyll-a forecasting module
│   ├── config.py              # Configuration parameters
│   ├── data_preparation.py    # PACE/Sentinel-3 data download
│   ├── model.py               # CNN-LSTM architecture
│   ├── train.py               # Training script
│   ├── predict.py             # Generate forecasts
│   ├── models/                # Saved model checkpoints
│   └── data/                  # Training data (gitignored)
│
├── microcystin_detection/      # Microcystin detection module
│   ├── config.py              # Configuration parameters
│   ├── data_collection.py     # PACE data + GLERL labels
│   ├── granule_processing.py  # Extract spatial patches
│   ├── balance_training_data.py  # Class balancing
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training script
│   ├── predict.py             # Generate predictions
│   ├── utils.py               # Helper functions
│   ├── glrl-hab-data.csv      # In-situ measurements (ground truth)
│   ├── models/                # Saved model checkpoints
│   └── data/                  # Training data (gitignored)
│
├── combined_forecasting/       # Combined pipeline module
│   ├── config.py              # Pipeline configuration
│   ├── pipeline.py            # Sequential/end-to-end inference
│   ├── train_combined.py      # End-to-end training
│   └── forecast.py            # Multi-day forecasting
│
├── visualization/              # Visualization tools
│   ├── dashboard.py           # Interactive Dash web app
│   ├── plotting.py            # Plotting utilities
│   └── notebooks/             # Demonstration notebooks
│       ├── 01_data_exploration.ipynb
│       ├── 02_chla_forecasting_demo.ipynb
│       ├── 03_microcystin_detection_demo.ipynb
│       └── 04_combined_forecast_demo.ipynb
│
├── archive/                    # Archived code (old structure)
│   ├── old_notebooks/         # Original Jupyter notebooks
│   ├── old_scripts/           # Original utility scripts
│   ├── label_data/            # Species classification data
│   └── grid_search_results/   # Hyperparameter search results
│
├── data/                       # Shared data resources
│   ├── Archived_Forecast_Bulletins/  # NOAA HAB bulletins
│   └── cache/                 # Downloaded satellite data cache
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore patterns
└── README.md                 # This file
```

## Installation

```bash
# Clone repository
git clone https://github.com/j-b-cox/HAB-F-Capstone.git
cd HAB-F-Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up NASA Earthdata credentials (required for satellite data)
earthaccess login
```

## Quick Start

### 1. Train Microcystin Detection Model

```bash
cd microcystin_detection
python data_collection.py    # Download and process training data
python train.py              # Train the model
```

### 2. Train Chlorophyll Forecasting Model

```bash
cd chla_lstm_forecasting
python data_preparation.py   # Download time-series data
python train.py              # Train the model
```

### 3. Generate Combined Forecasts

```bash
cd combined_forecasting
python forecast.py --days-ahead 7 --start-date 2024-07-01
```

### 4. Launch Interactive Dashboard

```bash
cd visualization
python dashboard.py
```

Navigate to http://localhost:8050 to view predictions.

## Data Sources

- **Satellite Data**: NASA PACE Ocean Color Instrument (OCI) L2 data
  - 172 hyperspectral bands (340-890 nm)
  - 1 km spatial resolution
  - Daily coverage (cloud-permitting)
  
- **Ground Truth Labels**: NOAA GLERL in-situ water quality measurements
  - Particulate and dissolved microcystin concentrations
  - Extracted chlorophyll-a
  - Station locations across Lake Erie

## Models

### Microcystin Detection CNN

- **Input**: Spatial patches (3×3, 5×5, 7×7, or 9×9 pixels) of 172-band hyperspectral data
- **Architecture**: Dual-input CNN (patch features + contextual features)
- **Output**: Binary classification (microcystin present/absent) with probability
- **Training**: Balanced dataset with oversampling, multiple thresholds (0.1, 5, 10 µg/L)

### Chlorophyll-a Forecasting CNN-LSTM

- **Input**: Time series of spectral reflectance images (5-day window)
- **Architecture**: Stacked ConvLSTM2D layers with batch normalization
- **Output**: Predicted chlorophyll-a concentration map for next time step
- **Training**: Mean squared error loss on log-transformed concentrations

### Combined Forecasting System

- **Option 1**: Spectral forecasting → Toxin detection
- **Option 2**: Toxin map generation → Temporal toxin forecasting
- **Training**: End-to-end backpropagation through both models

## Performance Metrics

- **Spatial Accuracy**: F1-score, AUC-ROC for toxin detection
- **Temporal Accuracy**: MSE, MAE for multi-day forecasts
- **Coverage**: Valid pixel fraction across Lake Erie
- **Latency**: Time from satellite overpass to forecast generation

## Contributing

This is a research project. For questions or collaboration opportunities, please contact the repository owner.

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added]
```

## License

See LICENSE file for details.

## Acknowledgments

- NOAA Great Lakes Environmental Research Laboratory (GLERL) for in-situ data
- NASA Ocean Biology Processing Group for PACE satellite data
- NSF for funding support

---

**Note**: This is a refactored codebase. Original notebooks and scripts are preserved in the `archive/` directory.
