# Phase 3 Integration Analysis: Chlorophyll Forecasting Strategy

**Date:** 2025-11-14  
**Analysis:** End-to-end validation and integration options for Phase 3

---

## Current Situation

### Data Inventory

**Sentinel-3 Composites (CNN-LSTM/Images2/):**
- **Total Files:** 1,037 composite files
- **Date Range:** 2017-01-01 to 2025-06-30 (8.5 years)
- **Temporal Resolution:** 3-day composites
- **Format:** NumPy arrays, shape `(1, 93, 163, 22)`
  - Dimension 0: Time (always 1 for composites)
  - Dimension 1-2: Spatial (93 × 163 pixels, Lake Erie region)
  - Dimension 3: 22 bands (Sentinel-3 OLCI channels)
  - **Band 21:** Chlorophyll-a concentration (mg/m³)
  - **Range:** -0.0014 to 25.68 mg/m³ (sample from 2024-07-01)

**Phase 2 Microcystin Model:**
- **Status:** Trained and validated (84.5% accuracy, 0.9351 AUC)
- **Input:** PACE OCI spectral data (172 channels)
- **Output:** Binary microcystin risk classification (≥1.0 µg/L)
- **Deployment:** Production-ready, merged to main
- **Model Files:** Currently in microcystin_detection/models/ (empty)

---

## Problem Statement

You want to refactor Phase 3 such that **the chlorophyll forecasting model uses predictions from the Phase 2 microcystin model as training data**, rather than raw chlorophyll band values.

This creates an interesting integration challenge:

1. **Sentinel-3 composites** (2017-2025) contain chlorophyll band 21
2. **Phase 2 microcystin model** was trained on **PACE OCI** (172 channels)
3. **Sensor mismatch:** S3 has 22 bands, PACE has 172 bands
4. **No existing microcystin predictions** for the Sentinel-3 composite dates

---

## Option Analysis

### Option 1: Generate Microcystin Predictions for All S3 Composites ❌

**Approach:** Run Phase 2 model on all 1,037 S3 composites to generate microcystin risk maps.

**Pros:**
- Complete historical predictions (2017-2025)
- Unified data pipeline

**Cons:**
- **FATAL FLAW:** Sensor incompatibility
  - Phase 2 model expects 172 PACE channels
  - S3 composites only have 22 channels
  - Cannot directly apply PACE-trained model to S3 data
- Would require retraining Phase 2 on S3 data
- Massive computational cost (1,037 full granule predictions)

**Verdict:** ❌ **Not feasible** without major Phase 2 refactoring

---

### Option 2: Train Chlorophyll Forecasting on Raw Chlorophyll (Current Design) ✅

**Approach:** Keep Phase 3 as-is - forecast chlorophyll from chlorophyll time series.

**Pipeline:**
```
S3 Composite (t-4 to t) → ConvLSTM → Chlorophyll (t+1 to t+7)
    [Band 21 only]                       [Concentration mg/m³]
```

**Pros:**
- ✅ Straightforward implementation
- ✅ 8.5 years of training data (1,037 composites)
- ✅ Directly addresses chlorophyll forecasting
- ✅ No sensor compatibility issues
- ✅ Module already designed for this

**Cons:**
- Separate from microcystin detection (no direct integration)
- Chlorophyll ≠ microcystin (correlation but not identity)

**Verdict:** ✅ **Recommended for Phase 3 completion**

---

### Option 3: Combined Architecture for Phase 4 ✅✅ (BEST)

**Approach:** Complete Phase 3 as chlorophyll forecasting, then build Phase 4 to integrate both models.

**Two-Stage Integration:**

**Stage 1: Phase 3 (Chlorophyll Forecasting)**
```
S3 Composites → ConvLSTM → Future Chlorophyll
[2017-2025]      [Phase 3]    [t+1 to t+7]
```

**Stage 2: Phase 4 (Combined Forecasting)**
```
PACE Spectral → Phase 2 → Current MC Risk
              [84.5% acc]

Current Chlorophyll + Phase 3 → Future Chlorophyll
[from PACE chl_ocx]   [ConvLSTM]   [t+1 to t+7]

Future Chlorophyll → Empirical/ML → Future MC Risk
  [forecasted]         [Phase 4]      [probabilistic]
```

**Key Insight:** Use chlorophyll as a **proxy for microcystin temporal dynamics**

**Phase 4 Options:**

**4A: Chlorophyll → Microcystin Mapping**
- Train model: `Chlorophyll concentration → Microcystin risk`
- Use Phase 2 training data (1,067 samples with MC labels)
- Apply to forecasted chlorophyll from Phase 3

**4B: Direct Microcystin Forecasting**
- Train ConvLSTM on microcystin time series
- Requires historical MC measurements or predictions
- More data-intensive but potentially more accurate

**4C: Ensemble Approach**
- Combine 4A and 4B predictions
- Weight based on confidence intervals
- Provide uncertainty quantification

**Verdict:** ✅✅ **OPTIMAL** - Maintains modularity, leverages all existing work

---

### Option 4: Hybrid Input (Chlorophyll + Microcystin Features) 🟡

**Approach:** Train ConvLSTM with multi-channel input combining chlorophyll and derived features.

**Pipeline:**
```
S3 Composites → Feature Engineering → ConvLSTM → Future State
  [Band 21]     [Chlorophyll + derived]   [Phase 3]   [Multi-output]
```

**Derived Features (from chlorophyll alone):**
- Spatial gradients (bloom edges)
- Temporal derivatives (growth rate)
- Statistical features (mean, std in window)
- Bloom indicators (threshold exceedance)

**Pros:**
- Richer input representation
- Can capture bloom dynamics
- No sensor mismatch

**Cons:**
- More complex preprocessing
- Still doesn't directly use Phase 2 model
- Requires feature engineering experimentation

**Verdict:** 🟡 **Possible enhancement** but not necessary for Phase 3

---

## Recommended Path Forward

### **RECOMMENDATION: Option 3 (Two-Stage Integration)** ✅✅

**Rationale:**
1. **Sensor Reality:** S3 and PACE are different sensors - we cannot directly apply PACE models to S3 data
2. **Modularity:** Each phase focuses on one task well
3. **Data Availability:** 1,037 S3 composites provide excellent chlorophyll training data
4. **Future Integration:** Phase 4 can combine both models for comprehensive forecasting

---

## Implementation Strategy

### Phase 3: Chlorophyll Forecasting (Complete As-Is)

**Objective:** Forecast chlorophyll concentrations 1-7 days ahead

**Architecture:**
```python
Input: Sequence of 5 chlorophyll maps (t-4, t-3, t-2, t-1, t)
       Shape: (5, 93, 163, 2)  # 2 channels: chla + mask

Model: ConvLSTM2D(32) → BN → ConvLSTM2D(32) → BN → Conv2D(1)

Output: Predicted chlorophyll map (t+1)
        Shape: (93, 163, 1)

Multi-step: Autoregressive forecasting for t+2 to t+7
```

**Training Data:**
- **Source:** CNN-LSTM/Images2/ (1,037 S3 composites)
- **Dates:** 2017-01-01 to 2025-06-30
- **Sequences:** ~1,032 overlapping 5-frame sequences
- **Split:** 60% train (619) / 20% val (206) / 20% test (207)

**Validation Metrics:**
- MSE on log-transformed chlorophyll
- MAE in mg/m³ (denormalized)
- Spatial correlation with ground truth
- Temporal consistency (smooth forecasts)

**Outputs:**
- `best_model.keras`: Best ConvLSTM checkpoint
- `training_history.png`: Loss curves
- `forecast_maps/`: Sample 7-day forecasts
- `Phase3_validation_report.md`: Performance analysis

---

### Phase 4: Combined Microcystin Forecasting (Future Work)

**Objective:** Forecast microcystin risk using both models

**Option 4A: Chlorophyll-to-Microcystin Mapping**

```python
# Use Phase 2 training data (1,067 PACE samples with MC labels)
# Extract chlorophyll from PACE chl_ocx product
# Train classifier: chla → MC risk

Input: Chlorophyll concentration (from Phase 3 forecast)
Model: Simple MLP or threshold-based classifier
Output: Microcystin risk probability

Pipeline:
  PACE → Phase 3 → Future Chla → Classifier → Future MC Risk
```

**Training Approach:**
1. Extract chlorophyll from Phase 2's 1,067 PACE samples
2. Train classifier: `chla value → MC ≥1.0 µg/L (binary)`
3. Apply to Phase 3 forecasts

**Expected Performance:**
- Chlorophyll is a known MC proxy (correlation ~0.6-0.8)
- Won't match Phase 2's 84.5% accuracy (spectral has more info)
- But provides temporal forecasting capability

**Option 4B: Direct Microcystin Time Series Forecasting**

```python
# Requires historical MC predictions or measurements

Input: Sequence of MC risk maps (from Phase 2 applied to historical PACE)
Model: ConvLSTM2D (similar to Phase 3)
Output: Future MC risk maps

Challenge: Need PACE historical data (2024-present only)
```

**Option 4C: Ensemble Forecast**

```python
# Combine spectral detection + temporal forecasting

Ensemble:
  Weight 1: Phase 2 current detection (high accuracy, no forecast)
  Weight 2: Phase 3 → Chla → MC (temporal forecast, lower accuracy)
  Weight 3: Phase 4B direct MC forecast (if enough data)

Output: Probabilistic MC risk with confidence intervals
```

---

## Data Requirements Analysis

### Phase 3 (Recommended Path)

| **Requirement** | **Status** | **Notes** |
|-----------------|------------|-----------|
| S3 Composites | ✅ Available | 1,037 files, 2017-2025 |
| Chlorophyll Band | ✅ Band 21 | Range 0-26 mg/m³ |
| Temporal Coverage | ✅ Excellent | 8.5 years, 3-day resolution |
| Spatial Coverage | ✅ Lake Erie | 93×163 pixels |
| Preprocessing Code | ✅ Complete | `chla_lstm_forecasting/utils.py` |
| Model Architecture | ✅ Complete | `chla_lstm_forecasting/model.py` |
| Training Pipeline | ✅ Complete | `chla_lstm_forecasting/train.py` |

**Ready to train immediately!**

### Phase 4 (Future Integration)

| **Requirement** | **Status** | **Notes** |
|-----------------|------------|-----------|
| Phase 2 Model | ✅ Available | 84.5% accuracy, deployed |
| Phase 2 Training Data | ✅ Available | 1,067 PACE samples with MC labels |
| Phase 3 Model | ⏳ Pending | Will be available after training |
| PACE Historical Data | ⚠️ Limited | 2024-present only (~1.5 years) |
| Chla-MC Correlation | 🔬 Research | Literature ~0.6-0.8, need to validate |

**Requires Phase 3 completion first**

---

## Computational Cost Analysis

### Option 1: Generate MC Predictions for All S3 (NOT RECOMMENDED)

- **Files:** 1,037 composites
- **Model:** Phase 2 (inference only)
- **Time:** ~1-2 sec/granule × 1,037 = **~30-60 minutes**
- **Storage:** 1,037 prediction maps × ~1 MB = **~1 GB**
- **Problem:** ❌ **SENSOR MISMATCH** - Cannot use PACE model on S3 data

### Option 3: Train Phase 3, Then Integrate in Phase 4 (RECOMMENDED)

**Phase 3 Training:**
- **Sequences:** ~1,032 training samples
- **Epochs:** 50 (with early stopping)
- **Batch Size:** 4
- **Time:** ~10-15 minutes on GPU
- **Storage:** Models ~500 KB, plots ~5 MB

**Phase 4 Integration:**
- **Training:** Depends on approach (4A: minutes, 4B: 10-15 min, 4C: 20-30 min)
- **Total:** < 1 hour end-to-end

---

## Validation Strategy

### Phase 3 End-to-End Validation

**Test 1: Training Convergence**
```bash
python -m chla_lstm_forecasting.train \
  --data-dir CNN-LSTM/Images2 \
  --sensor S3 \
  --seq-len 5 \
  --epochs 50 \
  --limit 100  # Quick test with 100 files
```

**Expected:**
- Validation loss decreases
- Training completes without errors
- Model checkpoint saved

**Test 2: Full Training**
```bash
python -m chla_lstm_forecasting.train \
  --data-dir CNN-LSTM/Images2 \
  --sensor S3 \
  --epochs 50
```

**Expected:**
- ~1,032 sequences created
- ~619 training / 206 val / 207 test
- Converged model (<0.01 MSE target)

**Test 3: 7-Day Forecast**
```bash
# Use latest 5 composites to forecast 7 days ahead
python -m chla_lstm_forecasting.predict \
  --model best_model.keras \
  --data-files $(ls CNN-LSTM/Images2/composite_data_S3_2025-*.npy | tail -5) \
  --n-steps 7
```

**Expected:**
- Smooth forecast progression
- Realistic chlorophyll values (0-50 mg/m³)
- Spatial coherence maintained

---

## Deliverables

### Phase 3 (Immediate)

- [x] Module structure complete (config, utils, model, train, predict)
- [x] Smoke tests passing (5/5)
- [ ] **End-to-end training on S3 data**
- [ ] **Sample forecasts generated**
- [ ] **Validation report with metrics**
- [ ] **Comparison with original LSTM.py**
- [ ] **Integration test suite**

### Phase 4 (Future)

- [ ] Chlorophyll-to-MC classifier (Option 4A)
- [ ] Direct MC forecasting model (Option 4B, if data available)
- [ ] Ensemble forecasting system (Option 4C)
- [ ] Combined visualization dashboard
- [ ] Uncertainty quantification
- [ ] Operational deployment pipeline

---

## Recommended Next Steps

### Step 1: Complete Phase 3 Training (TODAY)

```bash
# Quick test (10 minutes)
python -m chla_lstm_forecasting.train \
  --data-dir CNN-LSTM/Images2 \
  --sensor S3 \
  --limit 100 \
  --epochs 20

# Full training (15-20 minutes)
python -m chla_lstm_forecasting.train \
  --data-dir CNN-LSTM/Images2 \
  --sensor S3 \
  --epochs 50
```

### Step 2: Generate Sample Forecasts

```bash
# 7-day forecast from latest data
python -m chla_lstm_forecasting.predict \
  --model best_model.keras \
  --data-files $(ls CNN-LSTM/Images2/composite_data_S3_2025-06-*.npy | tail -5) \
  --n-steps 7
```

### Step 3: Validate Against Original

```bash
# Compare with CNN-LSTM/LSTM.py outputs
# Document differences and improvements
```

### Step 4: Integration Testing

```bash
# Create test_phase3_integration.py
python test_phase3_integration.py
```

### Step 5: Documentation

- Update README with training examples
- Create Phase 3 completion report
- Document model performance metrics

### Step 6: Plan Phase 4

- Design chlorophyll-to-MC mapping
- Prototype ensemble architecture
- Define integration API

---

## Decision Matrix

| **Approach** | **Feasibility** | **Data Req** | **Accuracy** | **Integration** | **Recommendation** |
|--------------|-----------------|--------------|--------------|-----------------|-------------------|
| **Option 1:** MC predictions for all S3 | ❌ Low | High | N/A | N/A | ❌ Not viable (sensor mismatch) |
| **Option 2:** Raw chlorophyll only | ✅✅ High | Low | Medium | Low | ✅ Good for Phase 3 |
| **Option 3:** Two-stage (Chla → MC in Phase 4) | ✅✅ High | Medium | High | High | ✅✅ **RECOMMENDED** |
| **Option 4:** Hybrid features | 🟡 Medium | Low | Medium | Medium | 🟡 Possible enhancement |

---

## Conclusion

**The best path forward is Option 3: Complete Phase 3 as chlorophyll forecasting, then build Phase 4 for integration.**

**Why:**
1. ✅ **Sensor compatibility:** S3 data works directly for chlorophyll forecasting
2. ✅ **Data richness:** 1,037 composites over 8.5 years
3. ✅ **Modularity:** Each phase has a clear, focused objective
4. ✅ **Integration ready:** Phase 3 output (forecasted chla) can feed Phase 4 (MC prediction)
5. ✅ **Scientific validity:** Chlorophyll is a known microcystin proxy
6. ✅ **Immediate value:** Chlorophyll forecasts are useful standalone

**Next Action:** Run Phase 3 training on full S3 dataset (15-20 min), validate outputs, then proceed to Phase 4 design.

---

**Prepared by:** GitHub Copilot  
**Date:** 2025-11-14  
**Status:** Ready for implementation
