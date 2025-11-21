# Phase 5: MC Forecasting ConvLSTM - Implementation Plan

**Status:** Ready to Begin  
**Date:** November 21, 2025  
**Prerequisites:** ✅ Complete (317 MC probability maps, 2024 & 2025 bloom seasons)

---

## 🎯 Objective

Develop a temporal forecasting model for microcystin (MC) probability in Lake Erie using ConvLSTM architecture. The model will learn from 20 months of satellite-derived MC probability maps to predict future bloom conditions 1-3 days ahead.

---

## 📊 Dataset Summary

**Complete MC Probability Maps:** 317 temporal maps
- **Date Range:** March 7, 2024 - October 1, 2025 (20 months)
- **2024 Bloom Season:** 104 maps (June-October)
- **2025 Bloom Season:** 77 maps (June-October)
- **Temporal Density:** ~16 maps/month average
- **Spatial Resolution:** 84×73 pixels (1.2 km grid)
- **Data Format:** .npy files with .npz coordinate files

**Data Quality:**
- Mean MC probability: 0.499 ± 0.206
- Max MC probability: 0.844 ± 0.162
- Valid pixels per map: 2,139 ± 1,054
- Lake Erie boundary masking applied

---

## 🏗️ Architecture Design

### Model Type: ConvLSTM (Proven Success)
Mirror the chlorophyll forecasting architecture that achieved MSE=0.3965:
- **Layer 1:** ConvLSTM2D (32 filters, 3×3 kernel, return_sequences=True)
- **Layer 2:** BatchNormalization
- **Layer 3:** ConvLSTM2D (32 filters, 3×3 kernel, return_sequences=False)
- **Layer 4:** BatchNormalization
- **Layer 5:** Conv2D output (1 filter for MC probability)

### Input Configuration
- **Sequence Length:** 5-7 days (lookback window)
- **Forecast Horizon:** 1-3 days ahead
- **Input Shape:** (seq_len, 84, 73, 1) 
  - 1 channel for MC probability (vs 2 channels for chlorophyll + mask)
- **Output Shape:** (84, 73, 1)

### Training Configuration
- **Loss Function:** MSE (probability regression)
- **Optimizer:** Adam (lr=1e-5, proven effective)
- **Batch Size:** 16
- **Epochs:** 100 with early stopping (patience=10)
- **Validation:** 2024 data split (80/20 train/val)
- **Final Test:** Entire 2025 bloom season (out-of-sample)

---

## 📁 Module Structure

```
mc_lstm_forecasting/
├── __init__.py           # Package initialization
├── config.py             # Configuration (paths, hyperparameters)
├── model.py              # ConvLSTM architecture
├── train.py              # Training pipeline
├── predict.py            # Inference pipeline
├── utils.py              # Data loading, sequences, preprocessing
├── README.md             # Documentation
└── best_model.keras      # Saved production model (after training)
```

---

## 🔄 Data Pipeline

### 1. Data Loading (`utils.py`)

```python
def load_mc_sequences(data_dir, seq_len=5, forecast_horizon=1):
    """
    Load MC probability maps and create temporal sequences
    
    Args:
        data_dir: Path to MC_probability_maps/
        seq_len: Number of days in lookback window
        forecast_horizon: Number of days ahead to predict
    
    Returns:
        X: (n_samples, seq_len, H, W, 1) - input sequences
        y: (n_samples, H, W, 1) - target maps
        dates: List of date strings for each sequence
    """
```

**Key Features:**
- Sort maps chronologically by date
- Handle temporal gaps (not every day has data)
- Skip sequences with missing dates in lookback window
- Load corresponding .npz coordinate files
- Normalize probabilities (already in [0, 1] range)

### 2. Temporal Split Strategy

**Critical Requirement:** Prevent temporal leakage

```python
# Split by year and date
train_maps = [all 2024 maps]          # 242 maps (entire year)
val_maps = [2025 maps before Aug 1]   # ~45 maps (Jan-Jul, early bloom)
test_maps = [2025 maps from Aug 1 on] # ~30 maps (Aug-Oct, peak bloom)
```

**Rationale:** 
- 2024 provides complete training data (full bloom cycle)
- 2025 validation set includes bloom onset (June-July)
- 2025 test set includes peak bloom (August-September)
- Both validation and test contain bloom season data
- Tests model's ability to predict different bloom stages
- True out-of-sample validation (2025 unseen during training)

### 3. Sequence Creation

For each valid temporal window:
1. Get `seq_len` consecutive dates with data
2. Stack as input sequence: `[t-seq_len+1, ..., t]`
3. Get target `forecast_horizon` days ahead: `t + forecast_horizon`
4. Skip if any dates missing in sequence or target

**Expected Output:**
- 2024 training: ~200-220 sequences
- 2025 validation: ~35-40 sequences (early bloom)
- 2025 test: ~25-30 sequences (peak bloom)

---

## 🎓 Training Pipeline

### Step 1: Data Preparation
```bash
python -m mc_lstm_forecasting.train --prepare-data
```
- Load all 317 MC probability maps
- Create temporal sequences
- Split into train/val/test sets
- Save preprocessed data for faster iteration

### Step 2: Model Training
```bash
python -m mc_lstm_forecasting.train --epochs 100 --batch-size 16 --lr 1e-5
```
- Build ConvLSTM architecture
- Train on 2024 sequences
- Validate on 2024 holdout
- Save best model based on validation loss
- Early stopping (patience=10)

### Step 3: Model Evaluation
```bash
python -m mc_lstm_forecasting.predict --test --visualize
```
- Load best model
- Predict on 2025 bloom season
- Calculate metrics: MSE, MAE, correlation
- Generate comparison visualizations

---

## 📈 Expected Performance

### Baseline Expectations
Based on chlorophyll forecasting success (MSE=0.3965):

**Optimistic Scenario:**
- MC forecasting should achieve **similar or better** performance
- Simpler task (1 channel vs 2, direct probability regression)
- MSE target: 0.02-0.04 (probability scale)
- MAE target: 0.10-0.15 (probability scale)

**Why MC might be easier:**
- Direct probability regression (vs chlorophyll concentration)
- Single input channel (less complexity)
- Clear temporal patterns in bloom dynamics
- Good quality input data (ensemble predictions)

**Why MC might be harder:**
- Less training data (317 maps vs 1,032 chlorophyll samples)
- Larger temporal gaps (not daily coverage)
- Bloom events are episodic (not continuous)

### Success Criteria
✅ **Minimum Acceptable:** MAE < 0.20 (probability scale)  
✅ **Target Performance:** MAE < 0.15  
✅ **Excellent Performance:** MAE < 0.10  
✅ **Temporal Generalization:** 2025 test MAE within 20% of 2024 validation MAE

---

## 📊 Visualization & Analysis

### 1. Forecast Comparison Maps
Using `visualize_mc_maps.py` as foundation:
- Side-by-side: Predicted vs Actual
- Error maps (absolute difference)
- Time series of forecasts through bloom season
- Animation of temporal evolution

### 2. Performance Analysis
- Scatter plots: Predicted vs Actual (per pixel)
- Temporal error trends (error over time)
- Spatial error patterns (where does model struggle?)
- Bloom onset detection accuracy

### 3. Bloom Season Comparison
- 2024 bloom patterns
- 2025 bloom patterns
- Inter-annual variability
- Peak bloom timing and intensity

---

## 🚀 Implementation Steps

### Step 1: Module Setup (Day 1)
- [x] Create `mc_lstm_forecasting/` directory
- [ ] Copy structure from `chla_lstm_forecasting/`
- [ ] Create all module files
- [ ] Adapt `config.py` for MC parameters

### Step 2: Data Pipeline (Day 1-2)
- [ ] Implement `load_mc_sequences()` in `utils.py`
- [ ] Test data loading with sample dates
- [ ] Verify sequence creation logic
- [ ] Confirm temporal split prevents leakage

### Step 3: Model Architecture (Day 2)
- [ ] Adapt `build_convlstm_model()` for 1-channel input
- [ ] Test model building
- [ ] Verify parameter count (~100K parameters expected)

### Step 4: Training (Day 2-3)
- [ ] Implement `train.py` pipeline
- [ ] Run initial training on 2024 data
- [ ] Monitor validation performance
- [ ] Save best model

### Step 5: Evaluation (Day 3-4)
- [ ] Test on 2025 bloom season
- [ ] Calculate all metrics
- [ ] Generate visualizations
- [ ] Document findings

### Step 6: Analysis & Documentation (Day 4-5)
- [ ] Compare 2024 vs 2025 bloom patterns
- [ ] Create forecast animations
- [ ] Write completion report
- [ ] Update main README

---

## 📝 Deliverables

1. **Trained Model:** `mc_lstm_forecasting/best_model.keras`
2. **Training Report:** Metrics, loss curves, validation results
3. **Test Results:** 2025 bloom season performance analysis
4. **Visualizations:** 
   - Forecast comparison maps
   - Error analysis plots
   - Temporal animations
   - Bloom season analysis
5. **Documentation:** 
   - Module README
   - API documentation
   - User guide for forecasting pipeline
6. **Completion Report:** Phase 5 summary with all findings

---

## 🎯 Success Indicators

✅ Model trains without errors  
✅ Validation loss decreases during training  
✅ Test MAE < 0.20 on 2025 bloom season  
✅ Model generalizes to 2025 (unseen year)  
✅ Forecasts show realistic spatial patterns  
✅ Bloom onset/peak timing approximately captured  
✅ Ready for operational deployment

---

## 📚 References

- **Phase 3 Success:** ConvLSTM chlorophyll forecasting (MSE=0.3965)
- **Phase 4 Success:** Ensemble MC detection (94.4% accuracy)
- **Dataset:** 317 MC probability maps (March 2024 - October 2025)
- **Architecture:** Proven ConvLSTM with 2 layers, 32 filters
- **Validation Strategy:** Temporal split (2024 train, 2025 test)

---

**Next Action:** Create `mc_lstm_forecasting/` module structure and begin data pipeline implementation.

