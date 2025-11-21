# MC Forecasting ConvLSTM - Phase 5

**Temporal forecasting of microcystin (MC) probability in Lake Erie using satellite data**

---

## 🎯 Overview

This module implements a ConvLSTM (Convolutional Long Short-Term Memory) neural network to forecast microcystin probability in Lake Erie 1-3 days ahead. The model learns from 20 months of satellite-derived MC probability maps spanning both 2024 and 2025 bloom seasons.

### Key Features
- **Temporal forecasting:** Predict MC probability 1-3 days ahead
- **Proven architecture:** Mirror successful chlorophyll forecasting (MSE=0.3965)
- **Complete dataset:** 317 MC probability maps (March 2024 - October 2025)
- **Temporal validation:** Train on 2024, test on 2025 (unseen bloom season)
- **High-resolution:** 1.2 km spatial resolution across Lake Erie

---

## 📊 Dataset

### Source Data
- **Input:** MC probability maps from PACE satellite (Phase 4 ensemble predictions)
- **Total Maps:** 317 temporal snapshots
- **Date Range:** March 7, 2024 - October 1, 2025 (20 months)
- **2024 Bloom Season:** 104 maps (June-October)
- **2025 Bloom Season:** 77 maps (June-October)

### Data Quality
- Mean MC probability: 0.499 ± 0.206
- Max MC probability: 0.844 ± 0.162
- Valid pixels per map: 2,139 ± 1,054
- Spatial resolution: 1.2 km grid (84×73 pixels)

### Temporal Coverage
```
TRAINING SET (2024 - 242 maps):
  March 2024:    8 maps
  April 2024:   20 maps
  May 2024:     20 maps
  June 2024:    20 maps    ← Bloom onset
  July 2024:    19 maps    ← Early bloom
  August 2024:  25 maps    ← Peak bloom
  September 2024: 19 maps  ← Late bloom
  October 2024: 21 maps    ← Bloom decline
  November 2024: 12 maps
  December 2024: 11 maps
  January 2025:  11 maps

VALIDATION SET (2025 Jan-Jul - ~45 maps):
  February 2025:  6 maps
  March 2025:   19 maps
  April 2025:   14 maps
  May 2025:     15 maps
  June 2025:     8 maps    ← Bloom onset (validation)
  July 2025:    22 maps    ← Early bloom (validation)

TEST SET (2025 Aug-Oct - ~30 maps):
  August 2025:  21 maps    ← Peak bloom (test)
  September 2025: 25 maps  ← Late bloom (test)
  October 2025:  1 map     ← Bloom decline (test)
```

---

## 🏗️ Model Architecture

### ConvLSTM Design
Based on successful chlorophyll forecasting architecture:

```
Input: (sequence_length=5, height=84, width=73, channels=1)
   ↓
ConvLSTM2D (32 filters, 3×3 kernel, return_sequences=True)
   ↓
BatchNormalization
   ↓
ConvLSTM2D (32 filters, 3×3 kernel, return_sequences=False)
   ↓
BatchNormalization
   ↓
Conv2D (1 filter) - Output: (84, 73, 1)
```

### Parameters
- **Total Parameters:** ~100,000 (similar to chlorophyll model)
- **Sequence Length:** 5 days lookback
- **Forecast Horizon:** 1 day ahead
- **Loss Function:** MSE (mean squared error)
- **Optimizer:** Adam (lr=1e-5)

---

## 🚀 Usage

### Training

```bash
# Train model on 2024 data
python -m mc_lstm_forecasting.train --epochs 100 --batch-size 16

# Options:
#   --sequence-length: Lookback window (default: 5)
#   --forecast-horizon: Days ahead to predict (default: 1)
#   --learning-rate: Adam learning rate (default: 1e-5)
#   --batch-size: Training batch size (default: 16)
#   --epochs: Maximum epochs (default: 100)
#   --patience: Early stopping patience (default: 10)
```

### Prediction

```bash
# Generate forecasts for specific dates
python -m mc_lstm_forecasting.predict --dates 20250701 20250715 20250801

# Test on entire 2025 bloom season
python -m mc_lstm_forecasting.predict --test --visualize

# Options:
#   --dates: Specific dates to predict (YYYYMMDD format)
#   --test: Evaluate on 2025 test set
#   --visualize: Generate comparison plots
#   --output: Output directory for predictions
```

### Evaluation

```bash
# Calculate metrics on test set
python -m mc_lstm_forecasting.predict --test --metrics

# Metrics computed:
#   - MSE: Mean squared error
#   - MAE: Mean absolute error
#   - Correlation: Spatial correlation
#   - Bias: Mean prediction bias
```

---

## 📁 Module Structure

```
mc_lstm_forecasting/
├── __init__.py           # Package initialization
├── config.py             # Configuration and hyperparameters
├── model.py              # ConvLSTM architecture definition
├── train.py              # Training pipeline
├── predict.py            # Prediction and evaluation
├── utils.py              # Data loading and preprocessing
├── README.md             # This file
└── best_model.keras      # Saved production model (after training)
```

---

## 🔄 Training Pipeline

### Step 1: Data Preparation

```python
from mc_lstm_forecasting.utils import load_mc_sequences

# Load and create sequences
X, y, dates = load_mc_sequences(
    data_dir="data/MC_probability_maps",
    seq_len=5,
    forecast_horizon=1
)

# X shape: (n_samples, 5, 84, 73, 1)
# y shape: (n_samples, 84, 73, 1)
```

### Step 2: Temporal Split

```python
# Split by year and date (prevent temporal leakage)
train_2024 = sequences[dates.year == 2024]  # All 2024 (~200-220 sequences)

# Split 2025 into validation and test
val_2025 = sequences[(dates.year == 2025) & (dates < '2025-08-01')]  # Jan-Jul (~35-40 sequences)
test_2025 = sequences[(dates.year == 2025) & (dates >= '2025-08-01')]  # Aug-Oct (~25-30 sequences)
```

**Key Advantage:** Both validation and test sets contain bloom season data
- Validation: Early bloom onset (June-July 2025)
- Test: Peak bloom conditions (August-September 2025)
- This tests model's ability to predict different bloom stages

### Step 3: Model Training

```python
from mc_lstm_forecasting.model import build_mc_convlstm

# Build model
model = build_mc_convlstm(
    input_shape=(5, 84, 73, 1),
    filters_1=32,
    filters_2=32,
    learning_rate=1e-5
)

# Train with early stopping
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint]
)
```

### Step 4: Evaluation

```python
# Test on 2025 bloom season
test_loss, test_mae = model.evaluate(test_dataset)

# Generate predictions
predictions = model.predict(test_sequences)

# Visualize results
from visualize_mc_maps import plot_mc_map
for date, pred, actual in zip(test_dates, predictions, targets):
    plot_comparison(date, pred, actual)
```

---

## 📈 Expected Performance

### Baseline Targets

Based on chlorophyll forecasting success (MSE=0.3965):

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| MAE (probability) | < 0.20 | < 0.15 | < 0.10 |
| MSE (probability²) | < 0.05 | < 0.03 | < 0.02 |
| Correlation | > 0.60 | > 0.75 | > 0.85 |

### Success Criteria
✅ Model trains without errors  
✅ Validation loss decreases during training  
✅ Test MAE < 0.20 on 2025 bloom season  
✅ Model generalizes to 2025 (unseen year)  
✅ Forecasts show realistic spatial patterns  
✅ Bloom onset/peak timing approximately captured  

---

## 🔬 Technical Details

### Why ConvLSTM?

**Advantages for MC Forecasting:**
1. **Spatial-Temporal Learning:** Captures both spatial patterns (bloom distribution) and temporal dynamics (bloom evolution)
2. **Proven Success:** Same architecture achieved MSE=0.3965 for chlorophyll forecasting
3. **Preserves Spatial Structure:** Unlike flattened LSTM, maintains 2D lake geometry
4. **Handles Gaps:** Can learn despite irregular temporal sampling

### Temporal Validation Strategy

**Critical Requirement:** Prevent temporal leakage in time series

```
Training:   2024 ALL (242 maps)
Validation: 2025 Jan-Jul (~45 maps, includes early bloom)
Test:       2025 Aug-Oct (~30 maps, includes peak bloom)
```

**Why This Matters:**
- Time series data has autocorrelation
- Random shuffle would leak future → past
- Year-based split ensures true out-of-sample validation
- Both val and test contain bloom season data for realistic evaluation
- Tests model on different bloom stages (onset vs peak)

**Bloom Season Coverage:**
- Training (2024): Full bloom cycle Jun-Oct (104 maps)
- Validation (2025): Bloom onset Jun-Jul (~30 maps)
- Test (2025): Peak bloom Aug-Sep (~46 maps)

### Data Handling

**Temporal Gaps:**
Not every day has satellite data (clouds, satellite coverage). The pipeline:
1. Sorts all 317 maps chronologically
2. Only creates sequences where all dates exist
3. Skips sequences with missing data in lookback window
4. Results in ~200-220 training sequences from 242 maps

**Input Normalization:**
MC probabilities are already in [0, 1] range from ensemble predictions. No additional scaling needed.

---

## 🎨 Visualization

### Forecast Comparison

```python
# Compare predicted vs actual for specific date
from mc_lstm_forecasting.predict import visualize_forecast

visualize_forecast(
    date="20250715",
    model_path="mc_lstm_forecasting/best_model.keras",
    output="forecast_20250715.png"
)
```

Output: 3-panel figure
- **Left:** Actual MC probability (ground truth)
- **Center:** Predicted MC probability (1-day forecast)
- **Right:** Absolute error map

### Temporal Animation

```python
# Create animation of forecast evolution
from mc_lstm_forecasting.predict import create_forecast_animation

create_forecast_animation(
    start_date="20250601",
    end_date="20250831",
    output="bloom_2025_forecast.mp4"
)
```

---

## 📚 References

### Related Phases
- **Phase 3:** Chlorophyll forecasting ConvLSTM (MSE=0.3965)
- **Phase 4:** MC detection ensemble (94.4% accuracy)
- **Dataset Generation:** 317 MC probability maps (complete_missing_months.py)

### Key Papers
- ConvLSTM: Shi et al. (2015) - "Convolutional LSTM Network"
- HAB Forecasting: Stumpf et al. (2016) - "Forecasting HABs in Lake Erie"
- Satellite MC Detection: Sayers et al. (2024) - "PACE OCI for HAB monitoring"

### Data Sources
- **Satellite:** PACE OCI L2 AOP (NASA Earthdata)
- **Ground Truth:** NOAA Great Lakes HAB forecasts
- **Lake Geometry:** Natural Earth 10m lakes shapefile

---

## 🤝 Contributing

This module follows the same patterns as `chla_lstm_forecasting/`:
- Type hints for all functions
- Comprehensive docstrings
- Logging throughout pipeline
- Config-driven parameters
- Reproducible random seeds

---

## 📄 License

Part of the HAB-F Capstone project (MIT License)

---

## ✨ Acknowledgments

- **Dataset:** PACE satellite mission (NASA)
- **Architecture:** Based on chlorophyll forecasting success
- **Validation:** NOAA Great Lakes Environmental Research Laboratory

---

**Status:** 🚧 In Development  
**Last Updated:** November 21, 2025  
**Next Steps:** Complete data loading pipeline, train initial model
