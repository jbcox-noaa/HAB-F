# Direct Spectral Forecasting: Deep Investigation

**Date:** November 24, 2024  
**Objective:** Investigate replacing indirect MC probability forecasting with direct spectral-to-toxin forecasting  
**Current Approach:** ConvLSTM predicts future MC probability (output of CNN detector)  
**Proposed Approach:** Model directly predicts microcystin from spectral history

---

## Executive Summary

You've identified a key inefficiency: **we're training a model to predict the predictions of another model**, which adds unnecessary complexity and potential error propagation. This document investigates direct approaches that leverage temporal spectral patterns to forecast microcystin risk.

### Key Insight
The current pipeline is:
```
Spectral Data → [CNN Detector] → MC Probability Maps → [ConvLSTM] → Future MC Probability
```

The proposed pipeline would be:
```
Temporal Spectral Sequences → [Unified Model] → Future MC Risk
```

---

## Current System Analysis

### Phase 4: Microcystin Detection (Spatial CNN)

**Architecture:**
```python
# Dual-input CNN
Input 1: Patch (patch_size × patch_size × 172 channels + 1 mask)
  ↓
  Conv2D(32 filters, 1×1) + BN  # Spectral mixing
  ↓
  Conv2D(64 filters, 1×1) + BN  # More spectral features
  ↓
  Reshape → Dense(128) per pixel
  ↓
  GlobalMaxPool across space → 128-dim patch embedding

Input 2: Context (lat, lon, doy, month, hour, mean reflectances)
  ↓
  Dense(64) + Dropout

Merge both branches
  ↓
  Dense(64→32→4) with BN and Dropout
  ↓
  Output: Sigmoid (binary: MC > 0.1 µg/L)
```

**Key Characteristics:**
- **Input:** Single timestep snapshot
- **Patch sizes:** 3×3, 5×5, 7×7, 9×9 pixels
- **Spectral bands:** 172 (PACE OCI)
- **Context features:** Temporal (doy, month, hour) + Spatial (lat, lon) + Spectral (means)
- **Output:** Binary classification or probability
- **Training data:** ~5,000 samples from GLERL in-situ measurements

**Limitations for Forecasting:**
1. ❌ No temporal memory - each prediction is independent
2. ❌ Cannot learn bloom dynamics or progression patterns
3. ❌ Temporal features (doy, month) are categorical, not sequential
4. ❌ Requires running CNN first, then forecasting its output

### Phase 5/6: MC Probability Forecasting (ConvLSTM)

**Architecture:**
```python
# Dual-channel ConvLSTM
Input: (seq_len=5, H=84, W=73, channels=2)
  - Channel 0: MC probability (from CNN detector)
  - Channel 1: Validity mask
  ↓
  ConvLSTM2D(32 filters, 3×3 kernel, return_sequences=True)
  ↓
  BatchNorm + Dropout(0.4)
  ↓
  ConvLSTM2D(32 filters, 3×3 kernel, return_sequences=False)
  ↓
  BatchNorm + Dropout(0.4)
  ↓
  Conv2D(1 filter, 1×1 kernel)
  ↓
  Output: (H, W, 1) - Next day MC probability
```

**Key Characteristics:**
- **Input:** 5-day history of MC probability maps
- **Lookback:** 5 days
- **Forecast horizon:** 1 day ahead
- **Gap handling:** Hybrid temporal + spatial interpolation
- **Performance:** 74% improvement over baseline (MSE=0.0247)
- **Training data:** ~800 temporal sequences from 2024-2025

**Limitations:**
1. ❌ Depends on CNN detector being run first
2. ❌ Error propagation from CNN → LSTM
3. ❌ Missing data filled with heuristics (temporal/spatial interpolation)
4. ❌ No direct access to raw spectral information
5. ✅ Good spatial-temporal modeling (ConvLSTM strength)

---

## Proposed Approaches

### Option 1: Temporal CNN (3D Convolutions)

**Concept:** Extend the spatial CNN to include temporal dimension

```python
Input: (seq_len=5, patch_size, patch_size, 172 channels)
  ↓
  Conv3D(32 filters, kernel=(3,1,1))  # Temporal mixing
  ↓
  Conv3D(64 filters, kernel=(1,1,1))  # Spectral mixing per time
  ↓
  GlobalAveragePooling over time
  ↓
  Spatial processing (same as current CNN)
  ↓
  Output: MC probability for day 6
```

**Pros:**
- ✅ Directly uses raw spectral data
- ✅ Learns temporal-spectral patterns jointly
- ✅ Simpler pipeline (one model instead of two)
- ✅ Can leverage pre-trained spatial features from Phase 4

**Cons:**
- ❌ 3D convolutions are computationally expensive
- ❌ Requires much more memory (5 days × 172 channels × spatial)
- ❌ Limited temporal range (5-7 days max practical)
- ❌ Harder to handle missing data (need 5 full days)

**Training Data Challenge:**
- Need: Sequences of (5 spectral days → 1 toxin measurement)
- Have: ~200 measurement dates from GLERL
- **Problem:** Creates only ~195 sequences vs ~800 for current approach
- **Impact:** Severe data scarcity

### Option 2: Spectral LSTM (Sequence Model)

**Concept:** Treat each spatial patch as a temporal sequence

```python
For each patch (pixel or NxN region):
  Input: (seq_len=5, 172 channels + context)
  ↓
  LSTM(128 units, return_sequences=True)
  ↓
  LSTM(64 units, return_sequences=False)
  ↓
  Dense(32) → Dense(1)
  ↓
  Output: MC probability for that patch
```

**Pros:**
- ✅ Direct spectral → toxin prediction
- ✅ Standard LSTM handles temporal dependencies well
- ✅ Can handle variable sequence lengths
- ✅ Memory efficient compared to 3D conv

**Cons:**
- ❌ Processes each spatial location independently
- ❌ Loses spatial coherence (bloom spatial patterns)
- ❌ No communication between neighboring patches
- ❌ Would need to run thousands of times per map (84×73 = 6,132 patches)

### Option 3: Hybrid ConvLSTM with Spectral Input (RECOMMENDED)

**Concept:** Replace MC probability maps with spectral feature maps as ConvLSTM input

```python
# Stage 1: Spectral Encoder (spatial, per timestep)
Input: (patch_size, patch_size, 172 channels)
  ↓
  Conv2D(32 filters, 1×1) - Spectral compression
  ↓
  Conv2D(16 filters, 1×1) - Further compression
  ↓
  Output: (patch_size, patch_size, 16 features)

# Stage 2: Apply encoder to each timestep → Stack temporal sequence
Input sequence: [(day1 spectra), (day2 spectra), ..., (day5 spectra)]
  ↓
  [Encoder] applied to each day
  ↓
  Stacked: (seq_len=5, patch_size, patch_size, 16 features)

# Stage 3: Temporal-Spatial ConvLSTM
Input: (5, patch_size, patch_size, 16)
  ↓
  ConvLSTM2D(32 filters, 3×3, return_sequences=True)
  ↓
  BatchNorm + Dropout
  ↓
  ConvLSTM2D(32 filters, 3×3, return_sequences=False)
  ↓
  Conv2D(1 filter, 1×1)
  ↓
  Output: (patch_size, patch_size, 1) - MC probability map
```

**Pros:**
- ✅ Direct spectral → toxin prediction
- ✅ Preserves spatial coherence (ConvLSTM strength)
- ✅ Learns spectral-temporal patterns jointly
- ✅ Can leverage Phase 4's spectral feature learning
- ✅ Memory efficient (16 features vs 172 channels)
- ✅ Handles missing data in spectral space (NaN masking)
- ✅ More training data (full spatial maps, not just patch centers)

**Cons:**
- ❌ More complex than current pipeline
- ❌ Need to handle missing spectral data carefully
- ❌ Requires downloading raw PACE data for each day (large!)
- ❌ Training is slower (more parameters, more data per sample)

### Option 4: Multi-Horizon Transformer

**Concept:** Use attention mechanism to predict multiple forecast horizons

```python
# Spatial patch encoder (per timestep)
Input: (patch_size, patch_size, 172 channels)
  ↓
  Patch embedding (learned)
  ↓
  Position encoding (spatial x, y)
  ↓
  Output: Sequence of patch tokens

# Temporal transformer
Input: [Patch tokens from day 1, day 2, ..., day 5]
  ↓
  Temporal position encoding
  ↓
  Multi-head self-attention (learn temporal dependencies)
  ↓
  Feed-forward network
  ↓
  Decoder: Predict days 6, 7, ..., 13 (multi-horizon)
  ↓
  Output: Multiple future toxin probability maps
```

**Pros:**
- ✅ Multi-horizon forecasting (1-7 days) from single model
- ✅ Attention mechanism learns what spectral patterns matter
- ✅ Can handle irregular time gaps naturally
- ✅ State-of-the-art for sequence modeling

**Cons:**
- ❌ Very complex architecture
- ❌ Requires massive amounts of training data
- ❌ Computationally expensive
- ❌ Hard to interpret learned attention patterns
- ❌ Overkill for this problem scale

---

## Data Availability Analysis

### Current Training Data

**Phase 4 (MC Detection):**
- Source: GLERL in-situ measurements
- Dates: 200 measurement events (Apr 2024 - May 2025)
- Samples: ~5,000 (multiple stations per date, multiple patch sizes)
- Spatial: Patch extraction around measurement locations
- Label: Binary (MC > 0.1 µg/L)

**Phase 5/6 (MC Forecasting):**
- Source: MC probability maps generated by Phase 4 ensemble
- Dates: 290 days with satellite coverage (Apr 2024 - Nov 2024)
- Samples: ~800 sequences (depends on gap constraints)
- Spatial: Full lake coverage (84 × 73 pixels)
- Label: Next day probability map

### Required Data for Direct Spectral Forecasting

**Option 1 (3D CNN):**
- Need: GLERL date + 5 prior days of PACE spectra
- Available: ~195 sequences (200 dates - 5 lookback)
- **Data reduction: 96% fewer sequences than current!**

**Option 3 (Hybrid ConvLSTM with Spectral - RECOMMENDED):**
- Need: Any 6 consecutive days of PACE spectra
- Available: Can create sequences from ALL satellite coverage dates
- Coverage analysis:
  - 290 days of PACE data (Apr 2024 - Nov 2024)
  - With 5-day lookback, 1-day forecast = 285 possible sequences
  - With 3-day max gap tolerance ≈ 150-200 valid sequences
  - **Spatial samples: 150 sequences × (84×73) pixels = ~900,000 pixel sequences!**

**Key Insight:** Option 3 can use **unsupervised** learning on spatial-spectral patterns, then fine-tune on GLERL labels.

### Data Storage Requirements

**Current Approach:**
- MC probability maps: 290 days × 84 × 73 × 4 bytes ≈ **7 MB**
- Already computed and cached

**Option 3 (Spectral Input):**
- PACE spectra: 290 days × 84 × 73 × 172 channels × 4 bytes ≈ **3.0 GB**
- Need to download and cache full spectral data

**Storage Comparison:**
| Approach | Size | Notes |
|----------|------|-------|
| Current (MC maps) | 7 MB | Already have |
| Option 3 (Spectral) | 3.0 GB | Need to download |
| Incremental cost | +3 GB | Feasible with 8-16GB RAM |

---

## Architecture Comparison Matrix

| Feature | Current (2-stage) | Option 1 (3D CNN) | Option 2 (LSTM) | **Option 3 (Hybrid)** | Option 4 (Transformer) |
|---------|------------------|-------------------|-----------------|---------------------|----------------------|
| **Directness** | ❌ Indirect | ✅ Direct | ✅ Direct | ✅ Direct | ✅ Direct |
| **Spatial Coherence** | ✅ Excellent | ✅ Good | ❌ Poor | ✅ Excellent | ✅ Good |
| **Temporal Modeling** | ✅ Good | ⚠️ Limited | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Training Data** | ✅ 800 seq | ❌ 195 seq | ❌ 195 seq | ✅ **900k pixels** | ❌ Insufficient |
| **Computational Cost** | ✅ Low | ❌ Very High | ✅ Low | ⚠️ Medium | ❌ Very High |
| **Memory Usage** | ✅ Low (7 MB) | ❌ Very High | ✅ Low | ⚠️ Medium (3 GB) | ❌ Very High |
| **Missing Data** | ✅ Handled | ❌ Problematic | ⚠️ Tolerable | ✅ Handled | ✅ Handled |
| **Multi-Horizon** | ❌ Single day | ❌ Single day | ❌ Single day | ⚠️ Retrain needed | ✅ Native |
| **Interpretability** | ✅ High | ⚠️ Medium | ⚠️ Medium | ✅ High | ❌ Low |
| **Implementation** | ✅ Done | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Complex | ❌ Very Complex |

---

## Recommendation: Hybrid ConvLSTM with Spectral Input (Option 3)

### Why This Approach?

1. **Data Efficiency:** Leverages full spatial coverage → 900k pixel-sequences vs 195 samples
2. **Direct Prediction:** Eliminates CNN detector dependency and error propagation
3. **Spatial-Temporal:** Preserves ConvLSTM's proven strength in modeling bloom dynamics
4. **Proven Architecture:** Builds on successful Phase 2 ConvLSTM design
5. **Feasible Scope:** More manageable than transformer, less limiting than 3D CNN

### Detailed Architecture Design

```python
# ============================================================================
# STAGE 1: Spectral Feature Encoder (TimeDistributed)
# ============================================================================
# Applied independently to each timestep
# Compresses 172 spectral bands → 16 learned features

class SpectralEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 1×1 convolutions for spectral compression
        self.conv1 = Conv2D(64, (1,1), activation='relu', name='spec_conv1')
        self.bn1 = BatchNormalization(name='spec_bn1')
        self.conv2 = Conv2D(32, (1,1), activation='relu', name='spec_conv2')
        self.bn2 = BatchNormalization(name='spec_bn2')
        self.conv3 = Conv2D(16, (1,1), activation='relu', name='spec_conv3')
        
    def call(self, x):
        # x: (batch, H, W, 172 + 1 mask)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        return x  # (batch, H, W, 16)

# ============================================================================
# STAGE 2: Temporal-Spatial ConvLSTM
# ============================================================================

def build_spectral_convlstm_forecaster(
    input_shape=(5, 84, 73, 173),  # (seq_len, H, W, 172 + mask)
    encoder_filters=[64, 32, 16],
    convlstm_filters=[32, 32],
    dropout=0.4,
    learning_rate=1e-3
):
    """
    Build spectral-to-toxin forecasting model.
    
    Architecture:
        Raw PACE spectra (5 days) 
        → Spectral encoder (TimeDistributed)
        → ConvLSTM layers (temporal-spatial)
        → Conv2D decoder
        → MC probability map (day 6)
    """
    # Input: temporal sequence of spectral maps
    input_layer = Input(shape=input_shape, name='spectral_sequence')
    
    # TimeDistributed spectral encoder
    encoder = SpectralEncoder()
    x = TimeDistributed(encoder, name='spectral_encoder')(input_layer)
    # Now: (batch, 5, 84, 73, 16)
    
    # ConvLSTM layers for temporal-spatial modeling
    x = ConvLSTM2D(
        filters=convlstm_filters[0],
        kernel_size=(3,3),
        padding='same',
        return_sequences=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name='convlstm1'
    )(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(dropout, name='dropout1')(x)
    
    x = ConvLSTM2D(
        filters=convlstm_filters[1],
        kernel_size=(3,3),
        padding='same',
        return_sequences=False,  # Output single timestep
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name='convlstm2'
    )(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(dropout, name='dropout2')(x)
    
    # Decoder: Feature map → MC probability
    x = Conv2D(1, (1,1), activation='sigmoid', name='output')(x)
    
    model = Model(inputs=input_layer, outputs=x, name='spectral_mc_forecaster')
    
    # Compile with masked MSE loss
    model.compile(
        optimizer=Adam(learning_rate),
        loss=masked_mse_loss,  # From Phase 2
        metrics=[masked_mae_loss]
    )
    
    return model
```

### Training Strategy

**Phase 1: Unsupervised Pre-training**
```python
# Goal: Learn spectral feature representations
# Dataset: All PACE data sequences (no labels needed)
# Loss: Reconstruction loss (autoencoder style)

# Encoder-Decoder architecture
encoder = SpectralEncoder()
decoder = SpectralDecoder()  # Mirror of encoder

# Train to reconstruct input spectra
# Input: day t spectra → Encode → Decode → Reconstruct day t spectra
# This learns useful spectral features without needing toxin labels
```

**Phase 2: Supervised Fine-Tuning**
```python
# Goal: Predict MC toxin from spectral sequences
# Dataset: GLERL measurement dates with 5-day lookback
# Loss: Binary cross-entropy or MSE on MC probability

# Use pre-trained encoder weights
# Add ConvLSTM + decoder on top
# Fine-tune entire model on toxin prediction task
```

**Phase 3: Spatial Expansion**
```python
# Goal: Generate full-lake probability maps
# Dataset: All spatial locations with temporal sequences
# Loss: Self-supervised + semi-supervised

# Leverage spatial patterns learned from GLERL point measurements
# Expand to unmeasured locations using spatial coherence
```

### Data Preprocessing Pipeline

```python
def create_spectral_sequences(
    pace_data_dir: str,
    seq_len: int = 5,
    forecast_horizon: int = 1,
    max_gap_days: int = 3
):
    """
    Create temporal sequences of PACE spectral data.
    
    Returns:
        X: (N, seq_len, H, W, 173) 
           172 spectral bands + 1 validity mask
        y: (N, H, W, 1) - MC labels (if available)
        metadata: dates, locations, coverage stats
    """
    # 1. Load all PACE granules in date range
    pace_files = sorted(glob(f"{pace_data_dir}/PACE_*.nc"))
    
    # 2. Extract relevant spectral bands
    # Keep: 340-890nm range (all 172 OCI bands)
    # Quality: Filter by quality flags
    
    # 3. Spatial processing
    # Crop to Lake Erie bbox: (-83.5, 41.3, -82.45, 42.2)
    # Reproject to common grid (84 × 73)
    # Aggregate multiple overpasses per day (max/mean)
    
    # 4. Temporal processing
    # Sort chronologically
    # Identify valid sequences (gap <= max_gap_days)
    # Create (X, y) pairs where y = MC prob if GLERL date, else NaN
    
    # 5. Masking
    # Create validity mask channel
    # Mask: 1 where spectral data exists, 0 for missing/cloud/land
    
    return X, y, metadata
```

### Multi-Horizon Extension

To predict multiple days ahead (e.g., 1, 3, 7 days):

**Option A: Separate Models**
```python
model_1day = build_spectral_convlstm_forecaster(forecast_horizon=1)
model_3day = build_spectral_convlstm_forecaster(forecast_horizon=3)
model_7day = build_spectral_convlstm_forecaster(forecast_horizon=7)

# Train each independently
# Advantage: Optimized for specific horizon
# Disadvantage: 3× training time, 3× storage
```

**Option B: Multi-Output Model**
```python
# ConvLSTM → 3 parallel Conv2D heads
outputs = [
    Conv2D(1, (1,1), activation='sigmoid', name='day_1')(convlstm_output),
    Conv2D(1, (1,1), activation='sigmoid', name='day_3')(convlstm_output),
    Conv2D(1, (1,1), activation='sigmoid', name='day_7')(convlstm_output),
]

model = Model(inputs=input_layer, outputs=outputs)

# Advantage: Shared feature learning, single model
# Disadvantage: More complex training, potential interference
```

**Option C: Autoregressive (RECOMMENDED)**
```python
# Iterative forecasting: predict day+1, feed back as input
def multi_day_forecast(model, initial_sequence, days_ahead=7):
    predictions = []
    current_seq = initial_sequence.copy()
    
    for day in range(days_ahead):
        # Predict next day
        pred = model.predict(current_seq)
        predictions.append(pred)
        
        # Update sequence: drop oldest, add prediction
        current_seq = np.roll(current_seq, shift=-1, axis=1)
        current_seq[:, -1, :, :, :] = convert_prob_to_spectral(pred)
        # Note: Need learned inverse mapping (prob → spectra approx)
    
    return predictions

# Advantage: Single model, natural multi-horizon
# Disadvantage: Error accumulation over time
```

---

## Implementation Roadmap

### Phase 1: Data Preparation (1-2 days)
- [ ] Download full PACE spectral data for Lake Erie (Apr 2024 - Nov 2024)
- [ ] Create spatial grid (84 × 73) aligned with current MC probability maps
- [ ] Process PACE granules: crop, reproject, aggregate by day
- [ ] Create spectral sequences with gap handling
- [ ] Save to efficient format (HDF5 or TFRecord)
- [ ] Compute normalization statistics per spectral band

**Deliverables:**
- `data/spectral_sequences.h5` (3-4 GB)
- `data/spectral_stats.npz` (normalization params)
- `data/sequence_metadata.json` (dates, coverage, quality)

### Phase 2: Model Development (2-3 days)
- [ ] Implement `SpectralEncoder` class
- [ ] Implement `build_spectral_convlstm_forecaster()`
- [ ] Create custom data generator for spectral sequences
- [ ] Implement masked loss functions (adapt from Phase 2)
- [ ] Write training script with callbacks
- [ ] Create visualization tools for spectral features

**Deliverables:**
- `microcystin_forecasting/spectral_model.py`
- `microcystin_forecasting/spectral_data_loader.py`
- `microcystin_forecasting/train_spectral.py`

### Phase 3: Pre-training (1 day)
- [ ] Implement spectral autoencoder for unsupervised learning
- [ ] Train encoder on all PACE sequences (no labels)
- [ ] Validate learned spectral features (PCA, t-SNE visualization)
- [ ] Save pre-trained encoder weights

**Deliverables:**
- `models/spectral_encoder_pretrained.keras`
- Visualization of learned spectral features

### Phase 4: Supervised Training (2-3 days)
- [ ] Create training sequences aligned with GLERL measurements
- [ ] Fine-tune model on toxin prediction task
- [ ] Experiment with training strategies:
  - Option A: GLERL dates only (~195 sequences)
  - Option B: Semi-supervised (GLERL + spatial propagation)
  - Option C: Transfer from Phase 4 CNN features
- [ ] Hyperparameter tuning (learning rate, dropout, filters)
- [ ] Train with data augmentation and regularization

**Deliverables:**
- `models/spectral_mc_forecaster_best.keras`
- Training logs and curves
- Hyperparameter search results

### Phase 5: Evaluation & Comparison (1-2 days)
- [ ] Evaluate on test set (2025 data)
- [ ] Compare to current Phase 2 ConvLSTM:
  - MSE, MAE, RMSE
  - Spatial error patterns
  - Temporal accuracy over multiple days
  - Coverage (ability to predict where Phase 2 cannot)
- [ ] Analyze failure modes
- [ ] Create comprehensive visualizations

**Deliverables:**
- `SPECTRAL_FORECASTING_RESULTS.md`
- Comparison plots and error analysis
- Prediction examples (best, median, worst)

### Phase 6: Production Integration (1 day)
- [ ] Create inference pipeline
- [ ] Implement multi-day autoregressive forecasting
- [ ] Integrate with existing visualization dashboard
- [ ] Document API and usage examples
- [ ] Write deployment guide

**Deliverables:**
- `microcystin_forecasting/predict_spectral.py`
- Updated dashboard with spectral model option
- Deployment documentation

---

## Expected Performance

### Hypothesized Improvements Over Current Approach

| Metric | Current (Phase 2) | Expected (Spectral) | Improvement |
|--------|------------------|---------------------|-------------|
| Test MSE | 0.0247 | **0.015-0.020** | +20-40% |
| Coverage | 55.8% | **70-80%** | +15-25 pts |
| Error propagation | Yes (CNN→LSTM) | **None** | Eliminated |
| Training samples | 800 sequences | **195 + 900k pixels** | 1000× spatial |
| Interpretability | Medium | **High** | Spectral features |
| Computational cost | Low | **Medium** | 2-3× |

### Why Better Performance is Expected

1. **Direct Learning:** No intermediate CNN prediction step
   - Eliminates error propagation from CNN misclassifications
   - Model learns optimal spectral-temporal features for forecasting

2. **Richer Input:** Raw spectra vs. binary/probability maps
   - 172 spectral bands contain much more information than single probability
   - Can learn subtle spectral signatures predictive of bloom progression
   - Temporal changes in spectra (slopes, patterns) directly inform forecasts

3. **Spatial Training Data:** Pixel-level sequences
   - Current: 800 full-map sequences
   - Spectral: 900,000 pixel sequences (84×73 × 150 dates)
   - Can learn local bloom dynamics across diverse spatial contexts

4. **Better Missing Data Handling:** NaN masking in spectral space
   - Current: Gap-filling heuristics (temporal/spatial interpolation)
   - Spectral: Learn to handle missing bands/pixels during training
   - Network can learn which spectral features are redundant/critical

5. **Transfer Learning Potential:**
   - Pre-trained spectral encoder on reconstruction task
   - Can leverage external spectral datasets (ocean color, atmospheric correction)
   - Potential to adapt to other HAB species or water bodies

---

## Risks & Mitigation

### Risk 1: Insufficient Training Data

**Risk:** Only 195 GLERL-labeled sequences (vs 800 for current model)

**Mitigation:**
- Use semi-supervised learning (unlabeled spectral sequences)
- Spatial data augmentation (900k pixel samples)
- Transfer learning from Phase 4 CNN (pre-trained spectral features)
- Data augmentation: spectral perturbations, spatial crops, temporal jitter

### Risk 2: Computational Complexity

**Risk:** Training on full spectral data (3 GB) is slower than probability maps (7 MB)

**Mitigation:**
- Use mixed-precision training (float16)
- Batch processing and efficient data loading (TFRecord format)
- Cloud GPU resources if needed (Google Colab, AWS)
- Spectral band selection (could reduce 172 → 50-100 most informative bands)

### Risk 3: Spectral-to-Probability Mapping Complexity

**Risk:** Relationship between spectra and toxin may be highly non-linear

**Mitigation:**
- Deeper encoder (add more conv layers if needed)
- Residual connections to preserve spectral information
- Ensemble with Phase 4 CNN (combine direct + indirect predictions)
- Analyze spectral feature importance (which bands matter most)

### Risk 4: Missing Data More Problematic

**Risk:** Clouds/gaps affect raw spectra more than gap-filled probability maps

**Mitigation:**
- Learn robust masked loss function (from Phase 2)
- Train with aggressive missing-data augmentation
- Implement spectral infilling network (parallel encoder-decoder)
- Fall back to Phase 2 model when spectral coverage too low

### Risk 5: Model May Not Outperform Current Approach

**Risk:** Added complexity might not yield significant performance gains

**Mitigation:**
- **Modular design:** Keep current Phase 2 as fallback
- **Ensemble approach:** Combine spectral model + current model
- **Hybrid architecture:** Use both spectra AND probability maps as dual input
- **Cost-benefit analysis:** Only deploy if improvement > 10% and justified

---

## Decision Criteria: Should We Pursue This?

### GO Decision (Implement Spectral Approach)

If ANY of these conditions hold:
1. ✅ **Performance target:** >20% improvement over current MSE=0.0247
2. ✅ **Coverage improvement:** >10 percentage points better spatial coverage
3. ✅ **Interpretability:** Spectral features provide scientific insights into bloom dynamics
4. ✅ **Generalization:** Model successfully transfers to other HAB types or time periods
5. ✅ **Operational value:** Reduces dependency on two-stage pipeline

### NO-GO Decision (Stay with Current)

If ANY of these critical issues occur:
1. ❌ **Performance:** <5% improvement over current (not worth complexity)
2. ❌ **Data scarcity:** Cannot generate sufficient training samples
3. ❌ **Computational infeasibility:** Training takes >5 days or requires >32GB RAM
4. ❌ **Unreliable predictions:** High variance in forecast quality
5. ❌ **Missing data failure:** Cannot handle gaps as well as current interpolation

### DEFER Decision (Investigate Further)

If in between:
- Run pilot study with simplified architecture (fewer parameters)
- Test on small subset of data (2-3 months)
- Benchmark computational requirements
- Consult with domain experts on spectral feature interpretability

---

## Next Steps

### Immediate Actions (This Week)

1. **Prototype Data Loader:**
   - Download 1 month of PACE data (May 2024)
   - Process into spectral sequences
   - Verify format and compute statistics
   - **Time estimate: 4 hours**

2. **Build Minimal Viable Model:**
   - Simple spectral encoder (2 Conv2D layers)
   - Single ConvLSTM layer
   - Train on 1 month subset
   - **Time estimate: 6 hours**

3. **Preliminary Evaluation:**
   - Compare to current model on same dates
   - Calculate MSE, coverage, spatial errors
   - Visualize predictions side-by-side
   - **Time estimate: 2 hours**

4. **Go/No-Go Decision:**
   - Review results with stakeholders
   - Decide whether to proceed with full implementation
   - **Time estimate: 1 hour**

### If GO: Full Implementation (2-3 Weeks)

Follow the 6-phase roadmap detailed above.

**Total time estimate: 10-15 days**
**Resource requirements: 8-16GB RAM, GPU recommended**

---

## Conclusion

The proposed **Hybrid ConvLSTM with Spectral Input** offers a compelling alternative to the current two-stage approach:

✅ **Direct prediction** eliminates error propagation  
✅ **Richer input** (172 bands vs 1 probability)  
✅ **More training data** (900k pixel sequences)  
✅ **Better interpretability** (spectral features)  
✅ **Feasible scope** (builds on proven ConvLSTM)

**Recommendation:** Proceed with 1-week pilot study to validate approach, then make data-driven decision on full implementation.

The potential for 20-40% performance improvement and elimination of CNN dependency makes this investigation worthwhile, while the modular design ensures we can fall back to the current successful Phase 2 model if needed.

---

**Questions for Discussion:**

1. Is 3-4 GB of spectral data storage acceptable?
2. Should we prioritize performance or interpretability?
3. What is the minimum acceptable improvement over current MSE=0.0247?
4. Should we ensemble spectral + probability models for robustness?
5. Is 2-3 week timeline acceptable for full implementation?

