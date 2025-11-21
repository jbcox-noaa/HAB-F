# Phase 4: Combined Chlorophyll Forecasting + Microcystin Detection
## Integration Architecture Design

**Date:** November 17, 2025  
**Status:** 🔄 **IN DESIGN**

---

## Overview

Phase 4 integrates:
- **Phase 3:** Chlorophyll-a forecasting (ConvLSTM, 3-day ahead predictions)
- **Phase 2:** Microcystin detection (CNN classifier, PACE spectral data)

**Goal:** Generate 3-day ahead harmful algal bloom (HAB) risk forecasts by:
1. Forecasting chlorophyll distribution using Sentinel-3 time series
2. Using forecasted chlorophyll to predict microcystin risk
3. Producing spatial risk maps for Lake Erie

---

## Integration Strategy: Option 4A (Chlorophyll → Microcystin Mapping)

### Why Option 4A?

**Advantages:**
- ✅ Leverages existing 1,067 PACE samples with MC measurements
- ✅ Fast inference (no additional training needed)
- ✅ Well-understood uncertainty (from Phase 2 validation)
- ✅ Can deploy immediately

**Architecture:**
```
Historical Sentinel-3 Composites (t-15 to t-3 days)
              ↓
    Phase 3: ConvLSTM Forecaster
              ↓
Predicted Chlorophyll Map (t+3 days)
              ↓
    Extract Chlorophyll Values
              ↓
    Phase 2: CNN Classifier
              ↓
Microcystin Risk Prediction (≥1.0 µg/L)
              ↓
   Spatial Risk Map + Uncertainty
```

---

## Data Flow Design

### Input Stage

**Historical Data (Sentinel-3):**
- Last 5 composites (15 days of history)
- 3-day composites from CNN-LSTM/Images2/
- Format: (5, 93, 163, 2) - sequence, spatial, channels
- Channels: normalized chlorophyll + valid pixel mask

### Forecasting Stage (Phase 3)

**Phase 3 ConvLSTM Model:**
- Input: Last 5 Sentinel-3 composites
- Output: Predicted chlorophyll map (t+3 days)
- Format: (93, 163, 1) - spatial prediction
- Range: [-1, 1] normalized → convert to mg/m³

**Inverse Transform:**
```python
def inverse_transform_chlorophyll(normalized, max_chla=500.0):
    log_value = (normalized + 1) / 2 * np.log10(max_chla + 1)
    chla_mgm3 = 10**log_value - 1
    return np.clip(chla_mgm3, 0.001, max_chla)
```

### Detection Stage (Phase 2)

**Challenge:** Phase 2 expects PACE spectral data (19 bands), but Phase 3 only predicts chlorophyll

**Solution Options:**

#### Option A1: Simplified Approach (RECOMMENDED)
Use chlorophyll threshold directly from Phase 2 training data:
- Extract chlorophyll-microcystin relationship from 1,067 PACE samples
- Build lookup table or regression: chlorophyll → MC probability
- Apply to forecasted chlorophyll map

**Advantages:**
- Simple and fast
- Directly uses Phase 2's validated data
- No spectral reconstruction needed

**Implementation:**
```python
# From Phase 2 training data
def build_chla_mc_model(pace_samples):
    """Build chlorophyll → MC probability model from PACE data"""
    # Extract chlorophyll from chl_ocx band
    chla_values = pace_samples['chl_ocx']
    mc_labels = pace_samples['mc_binary']  # ≥1.0 µg/L
    
    # Train simple classifier (logistic regression or threshold)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(chla_values.reshape(-1, 1), mc_labels)
    
    return model

def predict_mc_from_chla(chla_map, chla_mc_model):
    """Predict MC risk from forecasted chlorophyll"""
    # Flatten spatial map
    chla_flat = chla_map.flatten()
    
    # Predict probabilities
    mc_probs = chla_mc_model.predict_proba(chla_flat.reshape(-1, 1))[:, 1]
    
    # Reshape to spatial map
    mc_risk_map = mc_probs.reshape(chla_map.shape)
    
    return mc_risk_map
```

#### Option A2: Spectral Reconstruction (Advanced)
Reconstruct PACE spectral bands from forecasted chlorophyll:
- Train empirical model: chlorophyll → 19 PACE bands
- Use Phase 2's existing CNN classifier on reconstructed spectra

**Advantages:**
- Uses full Phase 2 CNN classifier
- Captures spectral relationships

**Disadvantages:**
- More complex
- Adds uncertainty from reconstruction step
- Requires additional training

---

## Module Structure

```
combined_forecasting/
├── __init__.py              - Package initialization
├── config.py                - Configuration (paths, thresholds, etc.)
├── forecast_pipeline.py     - Main integration pipeline
├── chla_mc_model.py         - Chlorophyll → MC prediction model
├── visualization.py         - Spatial risk mapping
├── uncertainty.py           - Uncertainty quantification
├── validate.py              - End-to-end validation
└── README.md                - Usage documentation
```

---

## Implementation Plan

### Step 1: Extract Chlorophyll-Microcystin Relationship

**Input:** Phase 2's 1,067 PACE samples with MC measurements

**Process:**
1. Load PACE samples from `data/PACE/*.nc`
2. Extract chlorophyll from `chl_ocx` band
3. Extract MC measurements (binary ≥1.0 µg/L)
4. Build statistical model (logistic regression or threshold)
5. Validate model performance

**Output:** Trained chlorophyll → MC probability model

### Step 2: Build Forecast Pipeline

**Process:**
1. Load last 5 Sentinel-3 composites
2. Run Phase 3 ConvLSTM forecaster
3. Get predicted chlorophyll map (t+3 days)
4. Inverse transform to mg/m³
5. Apply chlorophyll → MC model
6. Generate spatial risk map

**Output:** Spatial MC risk forecast (93×163 grid)

### Step 3: Uncertainty Quantification

**Sources of Uncertainty:**
1. Phase 3 forecast uncertainty (from validation MSE)
2. Chlorophyll → MC model uncertainty (from Phase 2 data)
3. Spatial variability (from valid pixel mask)

**Quantification:**
```python
def estimate_uncertainty(chla_forecast, chla_mc_model):
    # 1. Forecast uncertainty (from Phase 3 validation)
    forecast_std = 12.28  # MAE from validation in mg/m³
    
    # 2. MC model uncertainty (from Phase 2 ROC-AUC)
    mc_model_auc = 0.845  # From Phase 2 validation
    
    # 3. Combined uncertainty
    total_uncertainty = compute_combined_uncertainty(
        forecast_std, mc_model_auc, chla_forecast
    )
    
    return total_uncertainty
```

### Step 4: Visualization

**Output Maps:**
1. **Current Conditions** (t=0): Latest Sentinel-3 chlorophyll
2. **Chlorophyll Forecast** (t+3): Predicted distribution
3. **MC Risk Map** (t+3): Probability of MC ≥1.0 µg/L
4. **Uncertainty Map**: Combined forecast + detection uncertainty

**Format:** Interactive HTML dashboard or static PNG files

### Step 5: Validation

**Historical Validation:**
1. Select dates with both:
   - Sentinel-3 composites (for forecasting)
   - PACE MC measurements (for ground truth)
2. Run forecast pipeline
3. Compare predictions vs actual MC measurements
4. Calculate performance metrics (precision, recall, F1)

**Validation Period:** May-October 2024 (overlapping PACE data)

---

## Configuration

```python
# combined_forecasting/config.py

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PHASE3_MODEL = BASE_DIR / "chla_lstm_forecasting" / "best_model.keras"
PHASE2_DATA = BASE_DIR / "data" / "PACE"
S3_COMPOSITES = BASE_DIR / "CNN-LSTM" / "Images2"

# Thresholds
MC_THRESHOLD = 1.0  # µg/L for binary classification
CHLA_THRESHOLD = 10.0  # mg/m³ typical bloom threshold
HIGH_RISK_PROB = 0.7  # Probability threshold for high risk

# Forecast
FORECAST_DAYS = 3  # Days ahead
SEQUENCE_LENGTH = 5  # Historical composites needed

# Visualization
RISK_COLORS = {
    'low': '#00FF00',      # Green
    'medium': '#FFFF00',   # Yellow
    'high': '#FF0000'      # Red
}
```

---

## Expected Performance

### Phase 3 Forecasting (Validated)

- Test MSE: 0.3965 (normalized)
- MAE: 12.28 mg/m³
- Coverage: Variable by season (7-50% valid pixels)

### Phase 2 Detection (Validated)

- Test Accuracy: 84.5%
- ROC-AUC: 0.845
- Precision: 0.80 (high MC)
- Recall: 0.88 (high MC)

### Combined System (Estimated)

**Optimistic Estimate:**
- Forecast Accuracy: ~85% (MAE 12.28 on ~10-20 mg/m³ mean)
- Detection Accuracy: ~84.5% (from Phase 2)
- Combined: ~72% (0.85 × 0.845)

**Realistic Estimate:**
- Account for forecast error propagation
- Combined: ~65-70% accuracy for 3-day MC risk

**Uncertainty Sources:**
1. Chlorophyll forecast error: ±12.28 mg/m³
2. MC detection error: ~15.5% (1 - 0.845)
3. Spatial coverage: Variable (7-50%)
4. Temporal alignment: 3-day lag

---

## Success Criteria

### Minimum Viable Product (MVP)

✅ **Functional Integration:**
- Phase 3 forecast runs successfully
- Chlorophyll → MC model makes predictions
- Spatial risk map generated

✅ **Validated Performance:**
- Combined accuracy ≥60% on historical data
- False alarm rate ≤30%
- Detection rate ≥70% for high-risk blooms

✅ **Usable Output:**
- Clear spatial risk visualization
- Uncertainty estimates included
- Automated pipeline (minimal manual intervention)

### Stretch Goals

- Real-time deployment (daily forecasts)
- Uncertainty quantification improvements
- Integration with NOAA operational systems
- Interactive web dashboard

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Design architecture | 1 day | ✅ Complete |
| 2 | Extract chlorophyll-MC model | 1 day | 🔄 Next |
| 3 | Build forecast pipeline | 2 days | ⏳ Pending |
| 4 | Uncertainty quantification | 1 day | ⏳ Pending |
| 5 | Visualization | 1 day | ⏳ Pending |
| 6 | Validation | 2 days | ⏳ Pending |
| 7 | Documentation | 1 day | ⏳ Pending |

**Total:** 9 days (1.5-2 weeks)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Chlorophyll-MC correlation weak | Medium | High | Use Phase 2 ROC curve (0.845 AUC proves correlation exists) |
| Forecast error too large | Low | Medium | Already validated (MAE=12.28 mg/m³ acceptable) |
| Spatial coverage insufficient | Medium | Medium | Document limitations, focus on high-coverage periods |
| Temporal misalignment | Low | Low | Use 3-day composites (matches forecast horizon) |

---

## Next Steps

1. ✅ **Design architecture** (COMPLETE)
2. 🔄 **Extract chlorophyll-MC relationship** from Phase 2 PACE data
3. ⏳ Build `combined_forecasting` module
4. ⏳ Implement forecast pipeline
5. ⏳ Validate end-to-end system

---

**Status:** Architecture design complete, ready to implement!  
**Recommendation:** Proceed with Option A1 (simplified chlorophyll → MC model)  
**Next Action:** Extract chlorophyll-MC relationship from Phase 2 PACE samples
