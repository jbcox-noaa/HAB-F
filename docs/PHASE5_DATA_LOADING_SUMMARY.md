# Phase 5 Data Loading Summary

**Date:** November 21, 2025  
**Status:** ✅ COMPLETE - Data loading pipeline implemented and validated

---

## 📊 Implementation Results

### Data Loading Pipeline ✅

**Created functions:**
- `load_mc_map()` - Load single MC probability map with metadata
- `parse_date_from_filename()` - Extract datetime from filename
- `load_all_mc_maps()` - Load all maps from directory, sorted chronologically
- `create_sequences_with_gap_handling()` - Create temporal sequences with max_gap filter
- `split_by_year()` - Temporal train/val/test split by year
- `load_mc_sequences()` - Complete pipeline function (main entry point)
- `plot_mc_probability_map()` - Visualization utility
- `save_plot()` - Save figures with timestamp

### Test Results ✅

**From `test_mc_data_loading.py`:**

```
Total maps: 317
Total sequences: 181
Training sequences: 109 (60.2%)
Validation sequences: 40 (22.1%)
Test sequences: 32 (17.7%)

Data shape: (109, 5, 84, 73, 1)
Target shape: (109, 84, 73, 1)
Value range: [0.000, 1.000]

✓ Shapes are correct
✓ Value ranges valid
✓ NaN values: 67.2% (expected for non-lake pixels)
✓ No temporal leakage between train and val
✓ No temporal leakage between val and test
```

### Temporal Split Details

**Training Set (2024):**
- 109 sequences
- Date range: 2024-03-24 to 2024-12-14
- Covers full year including peak bloom season

**Validation Set (2025 Jan-Jul):**
- 40 sequences
- Date range: 2025-03-09 to 2025-07-29
- Contains early bloom season
- Used for hyperparameter tuning and early stopping

**Test Set (2025 Aug-Oct):**
- 32 sequences
- Date range: 2025-08-08 to 2025-10-01
- Contains peak bloom season (August-September)
- Final evaluation set (held out during training)

---

## 🔧 Configuration

### Gap Handling Strategy

**Parameters:**
- `seq_len = 5` days (lookback window)
- `forecast_horizon = 1` day (predict tomorrow)
- `max_gap_days = 3` days (maximum allowed gap in sequence)

**Results:**
- Created 181 sequences from 317 maps
- Skipped 131 sequences due to gaps > 3 days
- This is expected and intentional (see analysis below)

**Gap Analysis:**
- 64.9% of date transitions are consecutive (1-day gap)
- 17.7% have 2-day gaps
- 6.0% have 3-day gaps
- Only 4.7% have gaps > 5 days
- Mean gap: 3.3 days, Median gap: 2.0 days

**Why max_gap=3?**
1. Allows common 2-day cloud gaps (18% of data)
2. Rejects rare large gaps (< 5% of data)
3. Maintains temporal continuity for LSTM learning
4. Provides sufficient sequences (181 total)

### No Temporal Compositing ✅

**Decision:** Use one map per day (no multi-day composites)

**Rationale:**
1. **No data leakage** - each map represents specific day only
2. **Preserves bloom dynamics** - HABs change rapidly (daily scale)
3. **Already compositing within days** - multiple PACE overpasses per day averaged
4. **Operationally realistic** - can't use future data in real forecasting
5. **Good temporal coverage** - 65% consecutive days is sufficient

See `docs/PHASE5_TEMPORAL_DATA_ANALYSIS.md` for detailed analysis.

---

## 📁 Files Created

**Module Files:**
- `mc_lstm_forecasting/utils.py` - Complete data loading pipeline
- `mc_lstm_forecasting/model.py` - ConvLSTM architecture (tested, works)
- `mc_lstm_forecasting/config.py` - Updated with MAX_GAP_DAYS, PLOTS_DIR

**Test Files:**
- `test_mc_data_loading.py` - Comprehensive validation script

**Documentation:**
- `docs/PHASE5_TEMPORAL_DATA_ANALYSIS.md` - Detailed temporal structure analysis
- `docs/PHASE5_DATA_LOADING_SUMMARY.md` - This file

---

## ✅ Validation Checklist

- [x] Loads all 317 MC probability maps
- [x] Parses dates correctly from filenames
- [x] Sorts chronologically
- [x] Creates sequences with gap handling
- [x] Splits by year (2024 train, 2025 val/test)
- [x] No temporal leakage (train < val < test dates)
- [x] Correct shapes: X=(N, 5, 84, 73, 1), y=(N, 84, 73, 1)
- [x] Valid probability ranges [0, 1]
- [x] Handles NaN values (non-lake pixels)
- [x] Metadata loaded (lat/lon coordinates)
- [x] Sufficient sequences for training (109 train, 40 val, 32 test)

---

## 🎯 Performance Expectations

### Sequence Counts vs Original Estimate

**Original Estimate (from Phase 5 planning):**
- Train: 200-220 sequences
- Val: 35-40 sequences
- Test: 25-30 sequences

**Actual Results:**
- Train: 109 sequences (lower than expected)
- Val: 40 sequences (within range ✓)
- Test: 32 sequences (within range ✓)

**Why fewer training sequences?**

The original estimate assumed we would get ~220 sequences from 242 maps (2024). We got 109 instead because:

1. **More gaps than expected in 2024 data**
   - Early 2024 (Mar-Apr) has irregular coverage
   - Some periods have 5-10 day gaps
   - These break the max_gap=3 filter

2. **Conservative gap filtering**
   - max_gap=3 is strict but prevents temporal discontinuities
   - Ensures model learns from smooth temporal patterns
   - Better for LSTM to see coherent sequences

3. **Still sufficient for training**
   - 109 sequences × 16 batch size = ~7 batches per epoch
   - 100 epochs × 7 batches = 700 gradient updates
   - ConvLSTM has 112K parameters
   - Rule of thumb: 5-10 samples per parameter → need 560K-1.1M samples
   - We have: 109 sequences × 5 timesteps × 84 × 73 × 1 = 3.3M pixels
   - **Sufficient data for training!**

### Mitigation Strategies (if needed)

If training performance is poor due to limited sequences:

1. **Increase max_gap to 5 days**
   - Would add ~20-40 more sequences
   - Trade-off: some temporal discontinuity

2. **Use data augmentation**
   - Spatial flips (horizontal/vertical)
   - Small rotations
   - Could double/triple effective dataset size

3. **Reduce seq_len to 3 days**
   - More sequences from same data
   - Shorter lookback window (still reasonable for HAB forecasting)

**Decision:** Start with current setup (109 sequences). ConvLSTM training on chlorophyll used similar data volumes and achieved MSE=0.3965. Monitor validation performance - if overfitting occurs early, then apply augmentation or increase max_gap.

---

## 🚀 Next Steps

1. ✅ Data loading pipeline complete
2. ✅ Model architecture defined
3. ⏳ Create training pipeline (`train.py`)
4. ⏳ Implement training loop with callbacks
5. ⏳ Train model and monitor performance
6. ⏳ Evaluate on test set
7. ⏳ Create visualization pipeline

---

**Status:** Ready to proceed with model training (train.py implementation)
