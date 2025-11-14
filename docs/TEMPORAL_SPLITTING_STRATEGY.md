# Temporal Data Splitting Strategy for HAB-F

## Problem: Preventing Data Leakage in Time-Series Forecasting

### Why Temporal Splitting is Critical

**Data Leakage** occurs when information from the future "leaks" into training data, resulting in unrealistically high validation performance that doesn't generalize to true forecasting scenarios.

**Traditional random splitting** (train/val/test) is **invalid** for time-series because:
1. A model could "see" measurements from Day 10 during training
2. Then be asked to "predict" Day 7 during validation
3. This violates causality and inflates performance metrics

### PACE Data Timeline Analysis

**Data Available:**
- PACE mission started: April 16, 2024
- GLERL measurements in PACE era: April 24, 2024 - May 19, 2025
- Total measurement dates: 24 unique dates
- Temporal span: ~13 months (400 days)

**2024 Dates (22 dates):**
```
2024-04-24    2024-07-08    2024-08-21    2024-09-23
2024-05-08    2024-07-17    2024-08-26    2024-10-01
2024-05-20    2024-07-22    2024-09-03    2024-10-08
2024-06-03    2024-07-29    2024-09-10    2024-10-17
2024-06-17    2024-08-05    2024-09-16    2024-10-24
2024-07-02    2024-08-12
```

**2025 Dates (2 dates):**
```
2025-04-28
2025-05-19
```

## Recommended Temporal Split Strategy

### Option A: Strict Chronological Split (Conservative)

```
Train:      2024-04-24 to 2024-08-12  (12 dates, ~50%)
Validation: 2024-08-21 to 2024-09-23  (6 dates,  ~25%)
Test:       2024-10-01 to 2025-05-19  (8 dates,  ~33%)
```

**Pros:**
- ✅ Zero data leakage
- ✅ Simulates real forecasting scenario
- ✅ Tests generalization to future blooms

**Cons:**
- ⚠️ Limited training data (only 12 dates)
- ⚠️ Seasonal bias (train on spring/summer, test on fall/spring)

### Option B: Stratified Temporal Split (Recommended)

**Strategy:** Ensure each split contains data from different parts of the bloom season

```
Train:      2024: Apr 24, May 20, Jun 17, Jul 08, Jul 29, 
                  Aug 12, Sep 10, Sep 23, Oct 08
            (9 dates across full season, ~38%)
            
Validation: 2024: May 08, Jun 03, Jul 22, Aug 26, Oct 01
            (5 dates across season, ~21%)
            
Test:       2024: Jul 02, Jul 17, Aug 05, Aug 21, Sep 03, 
                  Sep 16, Oct 17, Oct 24
            2025: Apr 28, May 19
            (10 dates, ~42%)
```

**Selection Rule:** 
- Every 3rd date → validation
- Every 5th date → test  
- Remaining → train
- Ensures temporal gaps prevent leakage

**Pros:**
- ✅ Better seasonal coverage in each split
- ✅ More balanced split sizes
- ✅ Still maintains temporal integrity (no future → past)

**Cons:**
- ⚠️ More complex to implement
- ⚠️ Slight risk if using multi-day windows

### Option C: Rolling Window Cross-Validation (Advanced)

**For model comparison only, not final training**

```
Fold 1:  Train: Apr-Jun  Val: Jul     Test: Aug
Fold 2:  Train: Apr-Jul  Val: Aug     Test: Sep
Fold 3:  Train: Apr-Aug  Val: Sep     Test: Oct
Fold 4:  Train: Apr-Sep  Val: Oct     Test: 2025
```

**Pros:**
- ✅ Maximum use of limited data
- ✅ Assesses temporal stability
- ✅ Good for hyperparameter tuning

**Cons:**
- ❌ Cannot be used for final model evaluation
- ❌ Complex to implement

## Addressing the Limited Data Challenge

### Problem: Only ~24 dates is very limited

**Strategies to Maximize Data Usage:**

1. **Data Augmentation**
   - Spatial flips (already implemented)
   - Jittering coordinates slightly
   - Temporal interpolation (careful!)

2. **Transfer Learning**
   - Pre-train on earlier years (2012-2023) with other sensors
   - Fine-tune on PACE data with strict temporal split

3. **Semi-Supervised Learning**
   - Use unlabeled satellite imagery to learn features
   - Only use labeled data for final classification

4. **Multi-Year Strategy** 🌟 **RECOMMENDED**
   - Include 2012-2023 GLERL data with Sentinel-3/MODIS imagery
   - Temporal split: Train on 2012-2023, validate on 2024, test on 2025
   - Then fine-tune PACE-specific model with Option B split

## Proposed Implementation

### Configuration Update

```python
# microcystin_detection/config.py

# Temporal split strategy
TEMPORAL_SPLIT_STRATEGY = "stratified_temporal"  # Options: "strict_chronological", "stratified_temporal"

# Strict chronological split
STRICT_SPLIT = {
    "train_end": "2024-08-12",
    "val_end": "2024-09-23",
    "test_end": "2025-05-19",
}

# Stratified temporal split (every 3rd for val, every 5th for test)
STRATIFIED_SPLIT = {
    "train_dates": [
        "2024-04-24", "2024-05-20", "2024-06-17", "2024-07-08",
        "2024-07-29", "2024-08-12", "2024-09-10", "2024-09-23",
        "2024-10-08",
    ],
    "val_dates": [
        "2024-05-08", "2024-06-03", "2024-07-22", "2024-08-26",
        "2024-10-01",
    ],
    "test_dates": [
        "2024-07-02", "2024-07-17", "2024-08-05", "2024-08-21",
        "2024-09-03", "2024-09-16", "2024-10-17", "2024-10-24",
        "2025-04-28", "2025-05-19",
    ],
}

# Multi-year strategy
USE_HISTORICAL_DATA = True  # Include pre-PACE data for pre-training
HISTORICAL_SENSORS = ["SENTINEL3", "MODIS"]
HISTORICAL_TRAIN_END = "2023-12-31"
PACE_FINETUNE_EPOCHS = 50
```

### Validation Function

```python
def validate_temporal_split(train_dates, val_dates, test_dates):
    """Ensure no temporal leakage in split."""
    train_max = max(train_dates)
    val_min = min(val_dates)
    val_max = max(val_dates)
    test_min = min(test_dates)
    
    # Check chronological ordering
    assert train_max < val_min, f"Train data ({train_max}) leaks into val ({val_min})"
    assert val_max < test_min, f"Val data ({val_max}) leaks into test ({test_min})"
    
    # Check for overlaps
    assert set(train_dates).isdisjoint(val_dates), "Train/val overlap"
    assert set(train_dates).isdisjoint(test_dates), "Train/test overlap"
    assert set(val_dates).isdisjoint(test_dates), "Val/test overlap"
    
    return True
```

## Recommendation Summary

**For Phase 2 (Microcystin Detection Refactoring):**

1. ✅ **Use Option B (Stratified Temporal Split)** for PACE-only model
   - Better balance between data usage and temporal integrity
   - Maintains seasonal diversity

2. ✅ **Document current data split in config**
   - Make it easy to experiment with different strategies

3. ✅ **Implement temporal split validation**
   - Automatic checks prevent accidental leakage

4. 🔄 **Consider multi-year strategy for Phase 2.5**
   - Use 2012-2023 + Sentinel-3 for pre-training
   - Fine-tune on PACE with strict temporal split
   - Best of both worlds: more data + temporal integrity

**For Phase 3 (Chlorophyll Forecasting):**
- Same temporal split strategy
- CRITICAL for LSTM: ensure sequence windows don't cross split boundaries

**For Phase 4 (Combined Model):**
- Use same split dates as individual models
- Evaluate at multiple forecast horizons (1, 3, 5, 7 days)

## Next Steps

1. Update `microcystin_detection/config.py` with temporal split configuration
2. Implement `data_collection.py` with temporal split logic
3. Add validation checks to training pipeline
4. Document split strategy in model metadata

---

**IMPORTANT REMINDER:**
⚠️ **Update GLERL data (glrl-hab-data.csv) BEFORE starting Phase 2** ⚠️

This will affect the temporal split and should be done first to ensure we're working with the most current data.

---

Last Updated: 2025-11-13
