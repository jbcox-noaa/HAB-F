# Phase 5: Temporal Data Structure Analysis

**Date:** November 21, 2025  
**Purpose:** Understand how MC probability maps relate to days and whether compositing is needed

---

## 📊 Current Data Structure

### One Map Per Day ✅

**Analysis Results:**
```
Total MC probability maps: 317
Unique dates: 317
Maps with multiple per day: 0
```

**Key Finding:** We have **exactly one MC probability map per date**. There are no duplicate dates.

### Temporal Distribution

**Gap Analysis:**
```
Mean gap between maps: 3.3 days
Median gap: 2.0 days
Max gap: 12 days

Gap Distribution:
  1 day gap:  205 times (64.9%)  ← Most common
  2 day gap:   56 times (17.7%)
  3 day gap:   19 times (6.0%)
  4 day gap:   14 times (4.4%)
  5 day gap:    7 times (2.2%)
  6+ day gap:  15 times (4.8%)
```

**What This Means:**
- We have data **most days** (65% consecutive)
- Small gaps (2-3 days) are common (24%)
- Large gaps (>5 days) are rare (5%)
- Missing days are due to cloud cover or satellite coverage

---

## 🔄 How Maps Are Created Currently

### Process Flow

**From `generate_mc_probability_maps.py`:**

1. **Group PACE granules by date**
   ```python
   date_groups = {}
   for granule_path in pace_files:
       date_str = extract_date_from_filename(filename)
       date_groups[date_str].append(granule_path)
   ```

2. **For each date with data:**
   - Multiple PACE overpasses may occur on same day
   - Each granule gets processed through ensemble (4 models)
   - **Predictions are averaged across all granules for that day**
   ```python
   aggregated = np.nanmean(pred_stack, axis=0)
   ```

3. **Result: ONE probability map per date**
   - Already composite if multiple overpasses
   - Single overpass if only one granule that day

### Current Compositing Strategy

**Within-Day Aggregation:** ✅ Already happening
- Multiple PACE granules on same day → averaged
- This is good! Reduces noise from single observations

**Cross-Day Compositing:** ❌ Not currently done
- Missing days remain missing
- No temporal interpolation or gap-filling

---

## 🤔 Should We Create Multi-Day Composites?

### Option 1: Keep As-Is (One Map Per Day)

**Pros:**
- ✅ Preserves temporal resolution
- ✅ No data leakage concerns
- ✅ Reflects actual day-to-day variability
- ✅ Good coverage already (65% consecutive days)
- ✅ ConvLSTM can handle irregular sequences

**Cons:**
- ⚠️ Some temporal gaps (1-5 days mostly)
- ⚠️ Fewer total sequences (skip gaps in lookback window)

**Implementation:**
```python
# During sequence creation
if any date in [t-4, t-3, t-2, t-1, t] is missing:
    skip_this_sequence()
```

### Option 2: Create 3-Day Composites

**Approach:**
- For each date, average maps from [t-1, t, t+1]
- Fills small gaps (1-2 days)
- Still centered on specific day

**Pros:**
- ✅ Reduces cloud contamination
- ✅ Fills 1-2 day gaps
- ✅ Smoother temporal series
- ✅ More robust predictions

**Cons:**
- ⚠️ **TEMPORAL LEAKAGE RISK** 🚨
- ⚠️ Uses future data (t+1) to predict future
- ⚠️ Reduces temporal resolution
- ⚠️ Masks true day-to-day variability

**Leakage Example:**
```
Training sequence: [Day 1, Day 2, Day 3, Day 4, Day 5] → Predict Day 6

If Day 5 composite uses Day 6 data:
  We're using Day 6 to predict Day 6! ⚠️
```

### Option 3: Create Backward-Looking Composites

**Approach:**
- For each date, average maps from [t-2, t-1, t]
- Only uses past data
- Centers on most recent observation

**Pros:**
- ✅ No future data leakage
- ✅ Reduces noise
- ✅ Fills gaps if t-1 or t-2 available
- ✅ Operationally realistic (would only have past data)

**Cons:**
- ⚠️ Still reduces temporal resolution
- ⚠️ May over-smooth bloom dynamics
- ⚠️ Complicates interpretation (what day is this?)

**Example:**
```python
# Composite for Day 10
composite_10 = mean([map_8, map_9, map_10])  # Uses 3 most recent days

# This could still introduce subtle leakage:
# If predicting Day 11 from sequence ending Day 10,
# the Day 10 composite already incorporates recent trends
```

---

## 🎯 Recommended Approach

### **Keep One Map Per Day (Option 1)** ✅

**Rationale:**

1. **No Leakage Risk**
   - Each map represents actual observations from that specific day
   - No mixing of temporal information
   - Clear temporal boundaries

2. **ConvLSTM Handles Gaps Well**
   - Can learn from irregular sequences
   - 65% consecutive days is sufficient
   - Models will learn to handle missing data patterns

3. **Preserves Bloom Dynamics**
   - Harmful algal blooms can change rapidly (daily)
   - Day-to-day variability is scientifically important
   - Forecasting models need to see this variability

4. **Sufficient Data Density**
   - 317 maps over 20 months
   - Average 1 map every 2-3 days
   - Enough temporal resolution for 5-day sequences

5. **Operational Realism**
   - In real forecasting, you only have data up to current day
   - Can't create 3-day composite that includes tomorrow
   - Model should learn from actual irregular data patterns

### Implementation Strategy

**During Data Loading:**

```python
def create_sequences_with_gap_handling(dates, maps, seq_len=5, max_gap=3):
    """
    Create sequences only when all dates in lookback window exist
    and gaps between dates are <= max_gap days
    """
    sequences = []
    
    for i in range(seq_len, len(dates)):
        # Get dates for this sequence
        seq_dates = dates[i-seq_len:i]
        
        # Check if all dates exist (no missing days in lookback)
        gaps = [(seq_dates[j] - seq_dates[j-1]).days 
                for j in range(1, len(seq_dates))]
        
        # Skip if any gap is too large
        if max(gaps) > max_gap:
            continue  # Skip this sequence
        
        # Valid sequence - use these maps
        X = maps[i-seq_len:i]  # Shape: (5, H, W, 1)
        y = maps[i]             # Shape: (H, W, 1)
        sequences.append((X, y, dates[i]))
    
    return sequences
```

**Parameters:**
- `seq_len=5`: 5-day lookback window
- `max_gap=3`: Allow up to 3-day gaps within sequence
- This balances temporal continuity with data availability

**Expected Results:**
- ~200-220 training sequences (from 242 2024 maps)
- ~35-40 validation sequences (from ~45 2025 Jan-Jul maps)
- ~25-30 test sequences (from ~30 2025 Aug-Oct maps)

---

## 🔬 Scientific Considerations

### Why Daily Resolution Matters for HABs

**Bloom Dynamics:**
- Wind events can move blooms in 1-2 days
- Bloom intensity can double in 2-3 days during exponential growth
- Toxin production can vary day-to-day
- Management decisions often need 1-3 day forecasts

**From Literature:**
- Stumpf et al. (2012): Lake Erie HABs show daily variability
- Microcystin concentrations can change rapidly (24-48 hours)
- Satellite daily observations are standard for HAB monitoring

### Comparison to Chlorophyll Forecasting (Phase 3)

**Phase 3 Approach:**
```python
# chla_lstm_forecasting used Sentinel-3 data
# Input: Composite images (3-day or 7-day composites)
# This was appropriate because:
#   - Chlorophyll changes slowly (weekly timescales)
#   - Focus on spatial patterns, not rapid changes
```

**Phase 5 (MC Forecasting) Differences:**
- Microcystin is more variable than chlorophyll
- Bloom events are episodic (days to weeks)
- Forecasting horizon is shorter (1-3 days vs weeks)
- Need to capture rapid changes

**Conclusion:** Daily resolution is more appropriate for MC forecasting than for chlorophyll.

---

## ⚠️ Data Leakage Analysis

### Types of Leakage to Avoid

**1. Direct Future Leakage** ❌
```python
# BAD: Using Day 6 data to predict Day 6
composite_5 = mean([day_4, day_5, day_6])
sequence = [comp_1, comp_2, comp_3, comp_4, comp_5]
target = day_6  # LEAKAGE! comp_5 already contains day_6
```

**2. Indirect Temporal Leakage** ⚠️
```python
# SUBTLE: Training on future to predict past
# If we shuffle sequences randomly instead of temporal split
train_seq_1 = [..., day_100] → day_101
test_seq_1 = [..., day_50] → day_51
# Model might learn patterns from day_100 that help predict day_51
```

**3. Spatial Leakage** (Not relevant here)
- We're using full Lake Erie for all sets
- No spatial held-out regions

### How Our Approach Avoids Leakage ✅

**Temporal Split:**
```
Train: All 2024 maps
Val:   2025 Jan-Jul maps (after all training)
Test:  2025 Aug-Oct maps (after validation)
```

**Sequence Creation:**
```python
# Each sequence uses only past observations
Sequence: [day_t-4, day_t-3, day_t-2, day_t-1, day_t]
Target:   day_t+1  # Predict 1 day ahead

# No map contains data from day_t+1
# No shuffling across years
```

**No Compositing:**
- Each day's map is independent
- No mixing of temporal windows
- Clear causal relationship: past → future

---

## 📈 Expected Sequence Counts

### With max_gap=3 days

**Training (2024):**
- 242 total maps
- Approximately 200-220 valid sequences
- Loss of ~20-40 sequences due to gaps >3 days

**Validation (2025 Jan-Jul):**
- ~45 total maps
- Approximately 35-40 valid sequences
- Similar gap pattern expected

**Test (2025 Aug-Oct):**
- ~30 total maps
- Approximately 25-30 valid sequences
- Peak bloom season has good coverage (Aug: 21 maps, Sep: 25 maps)

### Sensitivity to max_gap

| max_gap | Training Seq | Val Seq | Test Seq | Trade-off |
|---------|--------------|---------|----------|-----------|
| 1 day   | ~150-170     | ~25-30  | ~20-25   | Very strict, fewer sequences |
| 2 days  | ~180-200     | ~30-35  | ~23-27   | Balanced |
| **3 days** | **~200-220** | **~35-40** | **~25-30** | **Recommended** ✅ |
| 5 days  | ~220-240     | ~40-45  | ~28-32   | Risk of discontinuity |

**Recommendation: max_gap=3**
- Allows common 2-day gaps (17.7% of data)
- Rejects rare large gaps (only 5% of data)
- Maintains temporal continuity
- Provides sufficient training data

---

## ✅ Final Recommendation

### Data Structure: One Map Per Day (No Compositing)

**Implementation:**
1. ✅ Keep existing MC probability maps as-is
2. ✅ One map per date (already done)
3. ✅ Implement gap-aware sequence creation
4. ✅ Use max_gap=3 days for filtering
5. ✅ Document gap statistics in model training logs

**Code to Add in `utils.py`:**

```python
def load_mc_sequences(
    data_dir: Path,
    seq_len: int = 5,
    forecast_horizon: int = 1,
    max_gap_days: int = 3,
    train_year: str = "2024",
    val_end_date: str = "20250801",
    test_start_date: str = "20250801"
) -> tuple:
    """
    Load MC probability maps and create temporal sequences.
    
    Args:
        data_dir: Directory containing mc_probability_*.npy files
        seq_len: Number of days in lookback window
        forecast_horizon: Number of days ahead to predict
        max_gap_days: Maximum allowed gap between consecutive dates in sequence
        train_year: Year for training data
        val_end_date: End date for validation (YYYYMMDD)
        test_start_date: Start date for test set (YYYYMMDD)
    
    Returns:
        (train_X, train_y, train_dates,
         val_X, val_y, val_dates,
         test_X, test_y, test_dates)
    
    Notes:
        - Each map represents ONE day (no compositing)
        - Sequences with gaps > max_gap_days are skipped
        - Temporal split prevents data leakage
        - Maps already aggregate multiple PACE overpasses per day
    """
    pass
```

**Advantages:**
- ✅ No data leakage
- ✅ Preserves bloom dynamics
- ✅ Operationally realistic
- ✅ Sufficient data density
- ✅ Scientifically sound

**Accepted Trade-offs:**
- ⚠️ Some sequences skipped due to gaps (acceptable - still 200+ sequences)
- ⚠️ Model must learn to handle irregular sampling (good - reflects reality)

---

**Decision:** Proceed with one map per day, no additional compositing ✅  
**Next Step:** Implement `load_mc_sequences()` in `utils.py` with gap handling
