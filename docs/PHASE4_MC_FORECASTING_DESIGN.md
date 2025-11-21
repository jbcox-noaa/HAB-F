# Phase 4: Direct Microcystin Forecasting
## ConvLSTM Architecture for MC Probability Prediction

**Date:** November 17, 2025  
**Status:** 🔄 **IN DESIGN**  
**Approach:** Generate MC probability maps → Train ConvLSTM → Forecast MC directly

---

## Strategy Overview

Instead of forecasting chlorophyll and then estimating microcystin risk, we'll **forecast microcystin probability directly**:

```
Phase 2 (Existing)              Phase 4 (New)
─────────────────────────────   ─────────────────────────────────

PACE Spectral Data              Historical MC Probability Maps
      ↓                                  ↓
CNN Classifier                    ConvLSTM Forecaster
      ↓                                  ↓
MC Probability Map              Future MC Probability Map
  (single date)                      (t+1 prediction)
```

**Key Advantages:**

✅ **Direct MC forecasting** - no intermediate steps  
✅ **Leverages Phase 2's validated classifier** (84.5% accuracy)  
✅ **Proven architecture** - same ConvLSTM that works for chlorophyll  
✅ **Simpler data pipeline** - single channel (MC probability)  
✅ **Better interpretability** - directly predicts what we care about (MC risk)  

---

## Architecture Design

### Input Data: MC Probability Maps

**Source:** Phase 2 CNN classifier predictions on all PACE granules

**Format:**
- **Shape:** `(H, W, 1)` - single channel (MC probability)
- **Range:** `[0, 1]` - probability of MC ≥ 1.0 µg/L
- **Spatial:** Variable based on PACE coverage (typically ~93×163 for Lake Erie)
- **Temporal:** Daily maps for all available PACE dates (2024-04-16 to present)

**Storage:**
```
data/MC_probability_maps/
├── 20240416_mc_prob.npy  # Shape: (93, 163)
├── 20240417_mc_prob.npy
├── 20240418_mc_prob.npy
...
```

### Model Architecture: MC-ConvLSTM

**Same architecture as Phase 3 chlorophyll model, but:**
- Input: 1 channel (MC probability) instead of 2 channels (chla + mask)
- Output: 1 channel (predicted MC probability)
- No log transform (already 0-1 range)

```python
# Input shape: (sequence_length=5, H, W, 1)
Input: (5, 93, 163, 1)
    ↓
ConvLSTM2D(32 filters, 3×3, return_sequences=True, tanh)
    ↓
BatchNormalization
    ↓
Dropout(0.2)  # Regularization from Phase 3 success
    ↓
ConvLSTM2D(32 filters, 3×3, return_sequences=False, tanh)
    ↓
BatchNormalization
    ↓
Dropout(0.2)
    ↓
Conv2D(1 filter, 3×3, sigmoid)  # Sigmoid for probability output [0,1]
    ↓
Output: (93, 163, 1)  # Predicted MC probability map
```

**Key Differences from Phase 3:**

| Aspect | Phase 3 (Chlorophyll) | Phase 4 (Microcystin) |
|--------|----------------------|----------------------|
| Input channels | 2 (chla + mask) | 1 (MC probability) |
| Output activation | tanh (normalized -1 to 1) | **sigmoid** (probability 0 to 1) |
| Preprocessing | log10 transform + normalization | **None** (already 0-1) |
| Loss function | MSE | **Binary crossentropy** or MSE |
| Parameters | ~113,697 | **~57,000** (fewer due to 1 channel) |

### Training Configuration

**Leverage Phase 3 success:**
- Random seed: **42** (reproducibility)
- Learning rate: **1e-4** (proven optimal)
- Batch size: **16**
- Early stopping patience: **10 epochs**
- Dropout rate: **0.2** (prevents overfitting)
- Loss: **Binary crossentropy** (better for probabilities) or MSE

**Data Split:**
- Train: 60% of PACE dates
- Validation: 20%
- Test: 20%

---

## Implementation Plan

### Step 1: Generate MC Probability Maps from PACE Data

**Goal:** Process all PACE granules through Phase 2 CNN to create daily MC probability maps

**Process:**

1. **Identify all available PACE dates**
   ```python
   # Search for PACE granules
   pace_files = glob.glob("data/PACE/*/*.nc")
   dates = extract_dates_from_files(pace_files)
   ```

2. **For each date, run Phase 2 classifier**
   ```python
   from microcystin_detection.predict import predict_from_granule
   
   for date in sorted(dates):
       # Get PACE granule for date
       granule_path = get_granule_for_date(date)
       
       # Run Phase 2 CNN classifier
       mc_prob_map = predict_from_granule(
           granule_path,
           model_path="microcystin_detection/best_model.keras"
       )
       
       # Save as .npy file
       output_path = f"data/MC_probability_maps/{date}_mc_prob.npy"
       np.save(output_path, mc_prob_map)
   ```

3. **Validate spatial consistency**
   - Check all maps have same shape (93, 163) or consistent dimensions
   - Handle missing data (cloud cover, instrument gaps)
   - Apply quality control (valid pixel threshold)

**Expected Output:**
- ~200-300 daily MC probability maps (depending on PACE data availability)
- April 2024 - November 2025 coverage
- Each map: `(H, W)` array with values in [0, 1]

### Step 2: Create MC Forecasting Module

**Directory Structure:**

```
mc_lstm_forecasting/
├── __init__.py              - Package initialization
├── config.py                - Configuration (modified from Phase 3)
├── utils.py                 - Data loading, preprocessing
├── model.py                 - MC-ConvLSTM architecture
├── train.py                 - Training script
├── predict.py               - Forecasting script
├── evaluate.py              - Evaluation metrics
├── tests/                   - Unit tests
│   ├── test_data_loading.py
│   ├── test_model.py
│   └── test_predictions.py
├── models/                  - Saved models
│   ├── best_model.keras
│   └── final_model.keras
└── README.md                - Documentation
```

**Key Configuration Changes:**

```python
# mc_lstm_forecasting/config.py

# Data source
DATA_SOURCE = "MC_PROBABILITY"  # Not PACE or Sentinel-3

# Input configuration
N_CHANNELS = 1  # MC probability only (no mask needed)
INPUT_RANGE = (0, 1)  # Already probability range
PREPROCESSING = None  # No log transform needed!

# Model architecture  
OUTPUT_ACTIVATION = "sigmoid"  # For probability output
LOSS_FUNCTION = "binary_crossentropy"  # Better for probabilities

# Data paths
MC_PROB_DIR = BASE_DIR.parent / "data" / "MC_probability_maps"
```

### Step 3: Adapt Data Loading Pipeline

**Key Differences from Phase 3:**

1. **No log transform** - data already in [0, 1] range
2. **Single channel** - just MC probability (no mask needed)
3. **Handle missing dates** - PACE data has gaps
4. **Quality control** - filter maps with insufficient valid pixels

```python
# mc_lstm_forecasting/utils.py

def load_mc_probability_sequences(
    mc_prob_dir: Path,
    sequence_length: int = 5,
    min_valid_pixels: int = 6500
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load MC probability maps and create sequences.
    
    Returns:
        X: Input sequences (N, seq_len, H, W, 1)
        y: Target maps (N, H, W, 1)
        dates: Corresponding dates
    """
    # Load all MC probability maps
    mc_files = sorted(mc_prob_dir.glob("*_mc_prob.npy"))
    
    mc_maps = []
    dates = []
    
    for f in mc_files:
        mc_prob = np.load(f)
        
        # Quality control: check valid pixels
        valid_pixels = np.sum(~np.isnan(mc_prob))
        if valid_pixels < min_valid_pixels:
            continue
            
        # Handle NaN (cloud cover, etc.)
        mc_prob = np.nan_to_num(mc_prob, nan=0.0)
        
        mc_maps.append(mc_prob)
        dates.append(extract_date_from_filename(f.name))
    
    # Create sequences
    X, y, seq_dates = create_sequences(
        np.array(mc_maps),
        dates,
        sequence_length=sequence_length
    )
    
    # Add channel dimension
    X = X[..., np.newaxis]  # (N, seq_len, H, W, 1)
    y = y[..., np.newaxis]  # (N, H, W, 1)
    
    return X, y, seq_dates
```

### Step 4: Train MC Forecasting Model

**Training Script:**

```python
# mc_lstm_forecasting/train.py

def train_mc_forecasting_model():
    """Train ConvLSTM to forecast microcystin probability."""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load MC probability sequences
    X, y, dates = load_mc_probability_sequences(
        mc_prob_dir=config.MC_PROB_DIR,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    print(f"Loaded {len(X)} sequences from {len(dates)} dates")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Build model
    model = build_mc_convlstm_model(
        input_shape=(config.SEQUENCE_LENGTH, *X.shape[2:]),
        dropout_rate=0.2
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',  # Better for probabilities
        metrics=['mae', 'mse', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=100,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  MAE: {test_results[1]:.4f}")
    print(f"  MSE: {test_results[2]:.4f}")
    print(f"  AUC: {test_results[3]:.4f}")
    
    return model, history
```

**Expected Performance:**

Based on Phase 3 success and Phase 2's 84.5% accuracy:

- **Target MSE:** < 0.05 (for probability predictions)
- **Target MAE:** < 0.15 (probability error)
- **Target AUC:** > 0.80 (for binary classification)
- **Convergence:** Expect validation loss to improve beyond epoch 1 (thanks to dropout)

### Step 5: Validation Metrics

**Regression Metrics (Probability Prediction):**
- MSE: Mean squared error between predicted and actual probabilities
- MAE: Mean absolute error
- R²: Coefficient of determination

**Classification Metrics (Binary MC Risk):**
- Threshold at 0.5: High risk (MC ≥ 1.0 µg/L) vs Low risk
- Precision: Of predicted high-risk areas, how many are actually high-risk?
- Recall: Of actual high-risk areas, how many did we predict?
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under receiver operating characteristic curve

**Spatial Metrics:**
- Spatial correlation: How well does spatial pattern match?
- Coverage: Percentage of valid predictions (non-NaN)

### Step 6: Visualization

**Output Maps:**

1. **Historical MC Probability** (t-15 to t)
   - Show last 5 days of observations
   - Animate temporal evolution

2. **Predicted MC Probability** (t+1)
   - Single-day forecast
   - Highlight high-risk areas (>0.5)

3. **Uncertainty Map**
   - Based on model confidence
   - Spatial variation in predictions

4. **Risk Classification**
   - Binary map: High risk (red) vs Low risk (green)
   - Threshold at 0.5 probability

**Example Visualization Code:**

```python
def visualize_mc_forecast(historical_maps, predicted_map, dates):
    """Visualize MC probability forecast."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot last 5 historical maps
    for i, (mc_map, date) in enumerate(zip(historical_maps, dates[-5:])):
        ax = axes.flatten()[i]
        im = ax.imshow(mc_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f"Observed: {date}")
        ax.axis('off')
    
    # Plot prediction
    ax = axes.flatten()[5]
    im = ax.imshow(predicted_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title(f"Forecast: {dates[-1] + timedelta(days=1)}")
    ax.axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label='MC Probability')
    plt.tight_layout()
    plt.savefig('mc_forecast.png', dpi=150)
```

---

## Data Requirements

### PACE Data Availability

**Expected Coverage:**
- Start date: April 16, 2024 (PACE launch)
- End date: November 17, 2025 (present)
- Duration: ~19 months
- Expected images: ~200-300 (accounting for cloud cover, gaps)

**Quality Control:**
- Minimum valid pixels: 6,500 (same as Phase 3)
- Cloud cover threshold: <70%
- Spatial consistency: All maps same dimensions

**Sequence Creation:**
- Sequence length: 5 days (same as Phase 3)
- Prediction horizon: 1 day (next-day forecast)
- Expected sequences: ~150-250 (depending on gaps)

### Train/Val/Test Split

With ~200 sequences:
- Train: 120 sequences (60%)
- Validation: 40 sequences (20%)
- Test: 40 sequences (20%)

**Sufficient for training?** 
- Phase 3 used 619 sequences → achieved test MSE 0.3965
- With 150-250 sequences, expect similar or slightly higher error
- Dropout + regularization crucial for preventing overfitting

---

## Comparison: Chlorophyll Forecasting vs MC Forecasting

| Aspect | Phase 3 (Chlorophyll) | Phase 4 (Microcystin) |
|--------|----------------------|----------------------|
| **Data Source** | Sentinel-3 composites | PACE + Phase 2 CNN |
| **Input Channels** | 2 (chla + mask) | 1 (MC probability) |
| **Input Range** | [-1, 1] (normalized) | [0, 1] (probability) |
| **Preprocessing** | log10 + normalization | None (already 0-1) |
| **Output** | Chlorophyll mg/m³ | MC probability [0,1] |
| **Output Activation** | tanh | **sigmoid** |
| **Loss Function** | MSE | **Binary crossentropy** |
| **Training Samples** | 619 sequences | ~150-250 sequences |
| **Parameters** | 113,697 | **~57,000** (fewer) |
| **Temporal Coverage** | 2017-2025 (8.5 years) | 2024-2025 (19 months) |
| **Interpretability** | Indirect (chla → MC) | **Direct (MC risk)** |

---

## Advantages of This Approach

### 1. **Direct Forecasting**
- Predicts what we actually care about (MC risk)
- No intermediate chlorophyll step
- More interpretable for stakeholders

### 2. **Leverages Existing Work**
- Phase 2 CNN already validated (84.5% accuracy)
- Phase 3 architecture proven to work
- Dropout strategy prevents overfitting

### 3. **Simpler Data Pipeline**
- Single channel input (MC probability)
- No log transform or complex normalization
- Already in probability range [0, 1]

### 4. **Better Uncertainty Quantification**
- Direct probability predictions
- Can threshold at different risk levels (0.3, 0.5, 0.7)
- Clear interpretation: "70% chance of high MC"

### 5. **Operational Deployment**
- Run Phase 2 on latest PACE data → MC probability map
- Feed last 5 days into Phase 4 → tomorrow's MC forecast
- Simple, automated pipeline

---

## Challenges and Mitigation

### Challenge 1: Limited Temporal Coverage

**Issue:** PACE only available since April 2024 (~19 months)

**Mitigation:**
- Use aggressive data augmentation (spatial crops, rotations)
- Transfer learning from Phase 3 weights (similar architecture)
- Strong regularization (dropout 0.2, early stopping)
- Accept slightly higher error than Phase 3

### Challenge 2: Data Gaps

**Issue:** PACE has cloud cover gaps, missing dates

**Mitigation:**
- Flexible sequence creation (skip missing dates)
- Quality control (minimum valid pixels)
- Document temporal gaps in validation

### Challenge 3: Fewer Training Samples

**Issue:** ~150-250 sequences vs Phase 3's 619

**Mitigation:**
- Smaller model (1 channel → fewer parameters)
- Dropout prevents overfitting
- Cross-validation for robust evaluation
- Document limitations in uncertainty estimates

---

## Success Criteria

### Minimum Viable Product (MVP)

✅ **Functional Pipeline:**
- Generate MC probability maps from PACE data
- Train MC-ConvLSTM model successfully
- Make next-day forecasts

✅ **Validated Performance:**
- Test MSE < 0.10 (probability error)
- Test AUC > 0.75 (binary classification)
- Validation loss improves beyond epoch 1

✅ **Usable Output:**
- Clear spatial risk maps
- High-risk area identification
- Uncertainty estimates

### Stretch Goals

- Multi-day forecasts (t+3, t+7)
- Ensemble predictions (multiple models)
- Real-time operational deployment
- Integration with NOAA systems

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Generate MC probability maps | 2 days | 🔄 Next |
| 2 | Create mc_lstm_forecasting module | 1 day | ⏳ Pending |
| 3 | Adapt data loading | 1 day | ⏳ Pending |
| 4 | Train MC-ConvLSTM | 2 days | ⏳ Pending |
| 5 | Validation & metrics | 1 day | ⏳ Pending |
| 6 | Visualization | 1 day | ⏳ Pending |
| 7 | Documentation | 1 day | ⏳ Pending |

**Total:** 9 days (~1.5-2 weeks)

---

## Next Steps

1. 🔄 **Investigate PACE data availability**
   - Count available PACE granules (April 2024 - November 2025)
   - Check spatial coverage and quality
   - Estimate number of usable dates

2. ⏳ **Generate MC probability maps**
   - Run Phase 2 CNN on all PACE granules
   - Save as .npy files in `data/MC_probability_maps/`
   - Validate spatial consistency

3. ⏳ **Create mc_lstm_forecasting module**
   - Copy Phase 3 structure
   - Modify config for 1-channel input
   - Update model architecture (sigmoid output)

4. ⏳ **Train and validate**
   - Load MC sequences
   - Train MC-ConvLSTM
   - Evaluate performance

---

**Status:** Design complete, ready to implement!  
**Recommendation:** Start with PACE data inventory and MC map generation  
**Next Action:** Count available PACE granules and estimate dataset size
