# Phase 7 Data Preparation Improvements

## Issues Identified and Solutions

### 1. **Temporal Matching Window Too Restrictive**

**Problem**: Original code only matched GLERL measurements to PACE data from the **exact same day**. Given:
- Cloud cover blocks many satellite passes
- GLERL sampling is sparse (weekly at best)
- Water properties change slowly (microcystin persists for days)

This resulted in only 58 matched samples from 1,767 GLERL measurements.

**Solution**: Expanded temporal window to **±1 day** (3-day window total)

**Implementation**:
```python
temporal_window_days: int = 1  # ±1 day window

for day_offset in range(-temporal_window_days, temporal_window_days + 1):
    search_date = glerl_date + timedelta(days=day_offset)
    # Try to match with PACE data from search_date
```

**Expected Impact**: Could increase labeled samples from 58 to 100-150 (2-3x improvement)

**Rationale**:
- Microcystin concentrations in surface waters are relatively stable over 1-2 days
- Satellite revisit time is ~1-2 days for PACE
- Water masses don't mix completely in 24 hours
- Published studies use ±3 day windows for satellite-insitu matchups

---

### 2. **Spatial Matching Distance Too Permissive**

**Problem**: Original code allowed matching up to **5 km** between GLERL station and PACE pixel. At Lake Erie's latitude:
- 5 km ≈ 4-5 PACE pixels
- Different water masses, shoreline influence, depth variations
- Mixing, advection can cause spatial heterogeneity

**Solution**: Reduced to **1.2 km** (one pixel radius)

**Implementation**:
```python
max_spatial_dist_km: float = 1.2  # One pixel length
```

**Expected Impact**: May reduce matches slightly, but ensures co-location

**Rationale**:
- PACE pixel size is ~1 km
- GLERL stations are point measurements
- Water quality can vary significantly over 5 km in coastal/nearshore zones
- Tighter spatial tolerance = more trustworthy labels

---

### 3. **Temporal Gap Inconsistency in Sequences**

**Problem**: Original sequence creation allowed variable gaps:
- Sequence 1: [Day 0, Day 1, Day 2, Day 3, Day 4] → all 1-day gaps
- Sequence 2: [Day 0, Day 1, Day 4, Day 5, Day 6] → [1, 3, 1, 1] gaps
- Sequence 3: [Day 0, Day 3, Day 4, Day 7, Day 8] → [3, 1, 3, 1] gaps

**Why this is a problem**:
1. **LSTM/ConvLSTM assumes uniform timesteps** - the recurrent connections treat each step equally
2. **Physically inconsistent** - 1-day vs 3-day changes represent different temporal dynamics
3. **Training instability** - model sees both "tomorrow" and "3 days later" as "next timestep"
4. **Prediction ambiguity** - at inference, which temporal spacing should we use?

**Solution**: Enforce uniform 1-day gaps

**Implementation**:
```python
MAX_GAP_DAYS = 1  # Changed from 3
REQUIRE_UNIFORM_GAPS = True  # New flag

# In create_temporal_sequences()
if require_uniform_gaps and valid_sequence:
    if not all(g == 1 for g in gaps):
        valid_sequence = False
```

**Expected Impact**: Fewer sequences (maybe 10-15 instead of 25), but all consistent

**Tradeoff Analysis**:
- ❌ Lose some training sequences (cloud gaps excluded)
- ✅ Model learns consistent temporal dynamics
- ✅ Predictions are temporally aligned (always 1-day ahead)
- ✅ Can use regular RNN/LSTM without time-encoding complexity

**Alternative Approaches** (not implemented):
1. **Time delta as input**: Add gap length as extra feature → adds complexity
2. **Temporal positional encoding**: Like transformers → requires architecture change
3. **Multiple models**: Train separate models for 1-day, 2-day, 3-day forecasts → inefficient

---

### 4. **Additional PACE Granules from EarthAccess** (TODO)

**Problem**: Current dataset has 187 granules spanning 121 unique dates. Possible gaps:
- Incomplete downloads
- New data released since last download
- Filtering may have excluded valid granules

**Solution**: Use NASA's `earthaccess` API to:
1. Search for all PACE OCI L2 AOP granules intersecting Lake Erie
2. Download any missing granules
3. Process and add to dataset

**Implementation** (pseudocode):
```python
import earthaccess

# Authenticate
earthaccess.login()

# Search for PACE granules
results = earthaccess.search_data(
    short_name='PACE_OCI_L2_AOP',
    bounding_box=(-83.5, 41.3, -82.45, 42.2),  # Lake Erie
    temporal=('2024-03-01', '2025-11-27')
)

# Download missing granules
earthaccess.download(results, local_path='./data/')
```

**Expected Impact**: Could add 10-30 more granules, 5-15 more dates, 5-20 more labeled samples

---

## Summary of Changes

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|---------|
| Temporal window | Same day only | ±1 day (3-day window) | 2-3x more labeled samples |
| Spatial distance | 5.0 km | 1.2 km | Higher quality matches |
| Max gap in sequences | 3 days | 1 day | Consistent temporal modeling |
| Uniform gaps required | No | Yes | Removes ambiguity |

## Expected Outcomes

### Before Improvements
- **Labeled samples**: 58 (9 positive, 49 negative)
- **Sequences**: ~25 (variable gaps)
- **Temporal consistency**: Low (gaps vary 1-3 days)

### After Improvements
- **Labeled samples**: ~100-150 (estimated)
- **Sequences**: ~10-15 (all uniform 1-day gaps)
- **Temporal consistency**: High (all exactly 1-day spacing)
- **Class balance**: Likely similar ratio (~20-30% positive)

## Training Implications

### Phase 1 (Unsupervised Pre-training)
- **Impact**: Fewer but higher-quality sequences
- **Benefit**: Model learns consistent temporal patterns
- **Risk**: Less data → may need more augmentation

### Phase 2 (Supervised Fine-tuning)
- **Impact**: 2-3x more labeled samples
- **Benefit**: Less overfitting, better generalization
- **Risk**: Temporal offset may add noise (but ±1 day is reasonable)

### Phase 3 (Semi-supervised)
- **Impact**: Cleaner pseudo-label generation
- **Benefit**: Consistent temporal spacing for predictions
- **Risk**: Fewer unlabeled sequences to expand into

## Validation Strategy

To verify improvements:

1. **Check class balance maintained**:
   ```python
   # Should still be ~20-30% positive samples
   positive_ratio = matched['mc_binary'].sum() / len(matched)
   ```

2. **Verify temporal offsets are small**:
   ```python
   # Most should be 0 (same day), some ±1 day
   temporal_offsets.value_counts()
   ```

3. **Confirm spatial co-location**:
   ```python
   # All should be ≤1.2 km
   assert all(spatial_distances <= 1.2)
   ```

4. **Inspect sequence uniformity**:
   ```python
   # All gaps should be exactly 1 day
   for dates in input_date_lists:
       gaps = [days_between(dates[i], dates[i+1]) for i in range(len(dates)-1)]
       assert all(g == 1 for g in gaps)
   ```

## References

- Stumpf et al. (2016): "Satellite-derived cyanobacteria bloom detection" - uses ±3 day window
- Hu et al. (2019): "Chlorophyll-a validation" - uses ±1 day for in-situ matchups
- IOCCG Protocol Series: Recommends spatial distance ≤ pixel size for validation

## Files Modified

1. `spectral_mc_forecasting/config.py`:
   - `MAX_GAP_DAYS`: 3 → 1
   - Added `REQUIRE_UNIFORM_GAPS = True`

2. `spectral_mc_forecasting/data_preparation.py`:
   - `match_glerl_to_pace_spectra()`: Added temporal window search
   - `create_temporal_sequences()`: Added uniform gap enforcement
   - Updated metadata saved in NPZ files

## Next Steps

1. ✅ Run updated pipeline
2. ⏳ Verify improvements (check sample count, class balance)
3. ⏳ Download additional PACE granules via earthaccess
4. ⏳ Re-run pipeline with complete dataset
5. ⏳ Begin Phase 1 model training
