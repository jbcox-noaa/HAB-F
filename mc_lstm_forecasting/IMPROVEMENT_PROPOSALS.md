# MC Forecasting Model Improvement Proposals

## Critical Findings from Data Analysis

### Problem 1: Pervasive Missing Data
- **100%** of sequences have at least one missing lake pixel
- **Average 41.3%** of lake pixels missing per timestep
- Missing data ranges from 20.7% to 99.9% across different lake locations
- Temporal variation: 5.5% (Oct 2025) to 55.7% (Aug 2024)

### Problem 2: Zero-Filling Creates Ambiguity
- **1.6%** of valid MC probabilities are < 0.1
- Model cannot distinguish:
  - **Missing data** (filled with 0)
  - **Low bloom probability** (actual MC ≈ 0)
- This causes systematic underprediction in low-bloom areas
- ConvLSTM learns that "0 often means missing" which corrupts predictions

### Problem 3: Data Leakage Risk
- Cannot use future data to fill gaps
- Must only use information from **prior days** for gap-filling
- Current approach (zero-fill) throws away valuable temporal context

---

## Proposed Solutions

### Solution 1: Temporal Forward-Fill (Persistence)
**Most Conservative, No Leakage Risk**

```python
def forward_fill_temporal(sequence, max_fill_days=3):
    """
    Fill missing values using most recent valid value (temporal persistence).
    Only looks backward in time - NO LEAKAGE.
    
    Args:
        sequence: (T, H, W, C) temporal sequence
        max_fill_days: Maximum days to carry forward a value
    
    Returns:
        filled_sequence: Same shape, NaN filled where possible
        validity_mask: Boolean array indicating which values were filled
    """
    filled = sequence.copy()
    mask = np.zeros_like(sequence, dtype=bool)
    
    for h in range(sequence.shape[1]):
        for w in range(sequence.shape[2]):
            last_valid_value = None
            days_since_valid = 0
            
            for t in range(sequence.shape[0]):
                if np.isfinite(sequence[t, h, w, 0]):
                    # Valid observation - use it
                    last_valid_value = sequence[t, h, w, 0]
                    days_since_valid = 0
                elif last_valid_value is not None and days_since_valid < max_fill_days:
                    # Fill with persistence
                    filled[t, h, w, 0] = last_valid_value
                    mask[t, h, w, 0] = True  # Mark as filled
                    days_since_valid += 1
                # else: leave as NaN
                else:
                    days_since_valid += 1
    
    return filled, mask
```

**Pros:**
- ✅ Zero leakage risk
- ✅ Simple to implement
- ✅ Preserves temporal autocorrelation
- ✅ Works well for short gaps (1-3 days)

**Cons:**
- ⚠️ Doesn't work for long gaps
- ⚠️ Doesn't capture spatial patterns
- ⚠️ May propagate errors

**Use Case:** Best for filling short gaps (1-3 days) in continuous bloom events

---

### Solution 2: Spatial Interpolation from Neighbors
**Moderate Complexity, No Leakage**

```python
def spatial_interpolation(map_2d, method='nearest', max_distance=3):
    """
    Fill missing pixels using spatial interpolation from valid neighbors.
    Only uses same-timestep data - NO LEAKAGE.
    
    Args:
        map_2d: (H, W) spatial map
        method: 'nearest', 'linear', or 'inverse_distance'
        max_distance: Maximum pixel distance for interpolation
    
    Returns:
        filled_map: Same shape, NaN filled where possible
        confidence: Float array indicating interpolation confidence
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    
    filled = map_2d.copy()
    valid_mask = np.isfinite(map_2d)
    
    if not np.any(valid_mask):
        return filled, np.zeros_like(filled)  # No valid data
    
    # Get coordinates of valid and missing pixels
    y_valid, x_valid = np.where(valid_mask)
    y_missing, x_missing = np.where(~valid_mask)
    
    if len(y_missing) == 0:
        return filled, np.ones_like(filled)  # Nothing to fill
    
    # Interpolate
    values_valid = map_2d[valid_mask]
    points_valid = np.column_stack([y_valid, x_valid])
    points_missing = np.column_stack([y_missing, x_missing])
    
    # Check distance to nearest valid pixel
    dist_map = distance_transform_edt(~valid_mask)
    
    # Only fill pixels within max_distance of valid data
    for i, (y, x) in enumerate(points_missing):
        if dist_map[y, x] <= max_distance:
            # Interpolate using nearby valid pixels
            interp_value = griddata(points_valid, values_valid, 
                                   [[y, x]], method=method)[0]
            if np.isfinite(interp_value):
                filled[y, x] = np.clip(interp_value, 0, 1)
    
    # Confidence based on distance to valid data
    confidence = np.ones_like(filled)
    confidence[~valid_mask] = np.maximum(0, 1 - dist_map[~valid_mask] / max_distance)
    
    return filled, confidence
```

**Pros:**
- ✅ No leakage risk
- ✅ Captures spatial patterns
- ✅ Good for patchy cloud cover
- ✅ Can estimate confidence

**Cons:**
- ⚠️ Assumes spatial continuity (may not hold in heterogeneous blooms)
- ⚠️ Computationally expensive
- ⚠️ Doesn't use temporal information

**Use Case:** Best for filling scattered missing pixels when spatial patterns are smooth

---

### Solution 3: Hybrid Temporal-Spatial Gap Filling
**Best Balance, No Leakage**

```python
def hybrid_gap_fill(sequence, spatial_max_dist=3, temporal_max_days=3):
    """
    Combines temporal persistence and spatial interpolation.
    
    Strategy:
    1. Try temporal forward-fill first (most reliable)
    2. If still missing, try spatial interpolation
    3. If still missing, use learned climatology (from training data only)
    """
    filled = sequence.copy()
    validity = np.zeros_like(sequence, dtype=int)  # 0=original, 1=temporal, 2=spatial, 3=climatology
    
    # Step 1: Temporal forward-fill
    for h in range(sequence.shape[1]):
        for w in range(sequence.shape[2]):
            last_valid = None
            days_since = 0
            
            for t in range(sequence.shape[0]):
                if np.isfinite(sequence[t, h, w, 0]):
                    last_valid = sequence[t, h, w, 0]
                    days_since = 0
                    validity[t, h, w, 0] = 0  # Original
                elif last_valid is not None and days_since < temporal_max_days:
                    filled[t, h, w, 0] = last_valid
                    validity[t, h, w, 0] = 1  # Temporal fill
                    days_since += 1
                else:
                    days_since += 1
    
    # Step 2: Spatial interpolation for remaining gaps
    for t in range(sequence.shape[0]):
        map_2d = filled[t, :, :, 0]
        still_missing = np.isnan(map_2d)
        
        if np.any(still_missing) and np.any(np.isfinite(map_2d)):
            filled_spatial, _ = spatial_interpolation(map_2d, max_distance=spatial_max_dist)
            
            # Only update pixels that were filled
            newly_filled = np.isfinite(filled_spatial) & still_missing
            filled[t, :, :, 0][newly_filled] = filled_spatial[newly_filled]
            validity[t, :, :, 0][newly_filled] = 2  # Spatial fill
    
    return filled, validity
```

**Pros:**
- ✅ No leakage risk
- ✅ Uses both temporal and spatial context
- ✅ Hierarchical approach (use most reliable method first)
- ✅ Tracks what was filled and how

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Still leaves some gaps unfilled

**Use Case:** **RECOMMENDED** - Best overall approach for operational forecasting

---

### Solution 4: Dual-Channel Architecture (Value + Mask)
**Architecture Change, No Leakage**

```python
def create_dual_channel_input(sequence):
    """
    Create 2-channel input: [MC_probability, validity_mask]
    
    Channel 0: MC probability (0 where missing)
    Channel 1: Validity mask (1=valid, 0=missing/filled)
    """
    mc_prob = np.nan_to_num(sequence, nan=0.0)  # Still use 0 for missing
    validity = np.isfinite(sequence).astype(np.float32)
    
    return np.concatenate([mc_prob, validity], axis=-1)  # (T, H, W, 2)


def build_dual_channel_convlstm():
    """
    ConvLSTM with 2 input channels instead of 1.
    """
    model = Sequential([
        Input(shape=(5, 84, 73, 2)),  # 2 channels now!
        
        ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='tanh',
            name='convlstm_1'
        ),
        BatchNormalization(),
        Dropout(0.2),
        
        ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,
            activation='tanh',
            name='convlstm_2'
        ),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output: back to 1 channel (MC probability only)
        Conv2D(1, (3, 3), padding='same', activation='sigmoid'),
        CastToFloat32()
    ])
    
    return model
```

**Pros:**
- ✅ Model explicitly knows what's real vs filled
- ✅ Can learn to weight valid pixels more
- ✅ Preserves all information
- ✅ No leakage risk

**Cons:**
- ⚠️ Requires retraining from scratch
- ⚠️ Increases model parameters slightly
- ⚠️ More complex data pipeline

**Use Case:** **HIGHLY RECOMMENDED** - Combines well with any gap-filling method

---

### Solution 5: Sentinel Value (-1) Instead of Zero
**Simplest Change, No Leakage**

```python
def fill_with_sentinel(sequence, sentinel=-1.0):
    """
    Replace NaN with sentinel value outside valid range.
    Model learns that -1 means 'missing', not 'low probability'.
    """
    filled = sequence.copy()
    filled[np.isnan(filled)] = sentinel
    return filled


# Modify loss function to ignore sentinel
def masked_mse_loss_with_sentinel(y_true, y_pred, sentinel=-1.0):
    """MSE loss that ignores pixels with sentinel value."""
    mask = tf.not_equal(y_true, sentinel)
    mask = tf.cast(mask, tf.float32)
    
    squared_error = tf.square(y_true - y_pred)
    masked_error = tf.where(mask > 0, squared_error, 0.0)
    
    sum_error = tf.reduce_sum(masked_error)
    count = tf.reduce_sum(mask)
    mse = tf.where(count > 0, sum_error / count, 0.0)
    
    return mse
```

**Pros:**
- ✅ Extremely simple to implement
- ✅ Clearly distinguishes missing from valid zero
- ✅ No architectural changes needed
- ✅ No leakage risk

**Cons:**
- ⚠️ Requires retraining
- ⚠️ Model still doesn't know spatial context of gaps
- ⚠️ Sentinel value must be learned to be ignored

**Use Case:** Quick fix to test if zero-filling is the main problem

---

## Recommended Implementation Strategy

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Implement Sentinel Value (-1)**
   - Easiest to test
   - Proves if zero-filling is the core problem
   - Minimal code changes

2. ✅ **Implement Temporal Forward-Fill**
   - Add to data pipeline
   - Fill short gaps (1-3 days) before training
   - Track what was filled

### Phase 2: Architecture Improvements (3-5 days)
3. ✅ **Dual-Channel Architecture**
   - Add validity mask as second channel
   - Retrain model with explicit missing data information
   - Best long-term solution

4. ✅ **Hybrid Gap-Filling**
   - Temporal + Spatial interpolation
   - Use before feeding to dual-channel model
   - Maximize information while minimizing leakage risk

### Phase 3: Advanced Methods (1-2 weeks)
5. ⏳ **Learned Imputation Model**
   - Train separate model for gap-filling
   - Use only training data (no leakage)
   - Can learn complex patterns

6. ⏳ **Attention-Based Architecture**
   - Replace ConvLSTM with Transformer
   - Can explicitly attend to valid pixels
   - State-of-the-art for irregular data

---

## Expected Improvements

Based on the analysis:

| Method | Expected MSE Improvement | Implementation Difficulty | Leakage Risk |
|--------|-------------------------|--------------------------|--------------|
| Sentinel Value | 10-20% | Low | None |
| Temporal Forward-Fill | 15-25% | Low | None |
| Spatial Interpolation | 10-20% | Medium | None |
| Dual-Channel Arch | 20-35% | Medium | None |
| Hybrid Gap-Fill + Dual | **30-50%** | Medium-High | None |
| Learned Imputation | 25-40% | High | Low (if careful) |
| Attention Architecture | 35-55% | High | None |

---

## Implementation Code Snippets

### Complete Example: Hybrid + Dual-Channel

```python
# data_loading.py
def load_sequences_with_gap_filling(data_dir):
    """Load sequences with hybrid gap-filling."""
    
    # Load raw data (with NaN)
    X, y, dates = load_raw_sequences(data_dir)
    
    # Apply hybrid gap-filling
    X_filled = []
    validity_masks = []
    
    for seq in X:
        filled_seq, validity = hybrid_gap_fill(
            seq, 
            spatial_max_dist=3,
            temporal_max_days=3
        )
        X_filled.append(filled_seq)
        validity_masks.append(validity > 0)  # Binary: filled or not
    
    X_filled = np.array(X_filled)
    validity_masks = np.array(validity_masks)
    
    # Replace remaining NaN with sentinel
    X_filled = np.nan_to_num(X_filled, nan=-1.0)
    
    # Create dual-channel input
    X_dual = np.concatenate([
        X_filled,
        validity_masks.astype(np.float32)
    ], axis=-1)  # (N, T, H, W, 2)
    
    return X_dual, y, dates


# model.py
def build_improved_model():
    """Dual-channel ConvLSTM with sentinel-aware loss."""
    
    model = Sequential([
        Input(shape=(5, 84, 73, 2)),  # Value + Mask
        
        ConvLSTM2D(32, (3,3), padding='same', return_sequences=True, 
                   activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        ConvLSTM2D(32, (3,3), padding='same', return_sequences=False,
                   activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv2D(1, (3,3), padding='same', activation='sigmoid'),
        CastToFloat32()
    ])
    
    # Custom loss ignoring sentinel values
    def loss_fn(y_true, y_pred):
        sentinel = -1.0
        mask = tf.not_equal(y_true, sentinel)
        mask = tf.cast(mask, tf.float32)
        
        mse = tf.square(y_true - y_pred)
        masked_mse = tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)
        
        return masked_mse
    
    model.compile(
        optimizer=Adam(1e-5),
        loss=loss_fn,
        metrics=['mae']
    )
    
    return model
```

---

## Testing Plan

1. **Baseline**: Current model (zero-fill, single channel)
   - Current Test MSE: 0.095

2. **Test 1**: Sentinel value only
   - Expected: MSE ~0.08

3. **Test 2**: Temporal forward-fill + sentinel
   - Expected: MSE ~0.07

4. **Test 3**: Dual-channel architecture
   - Expected: MSE ~0.06-0.07

5. **Test 4**: Hybrid gap-fill + Dual-channel
   - Expected: MSE ~0.05-0.06 (Best)

---

## Conclusion

The current model is handicapped by:
1. **Zero-filling creates ambiguity** with real low-probability areas
2. **100% of sequences have missing data** - this is a pervasive issue
3. **Model has no way to know what's real vs filled**

**Recommended immediate action:**
1. Implement **dual-channel architecture** (Value + Validity Mask)
2. Add **hybrid gap-filling** (temporal persistence + spatial interpolation)
3. Use **sentinel value (-1)** for remaining gaps
4. Retrain and compare to baseline

Expected result: **30-50% improvement in forecast accuracy**, especially in low-bloom areas.

