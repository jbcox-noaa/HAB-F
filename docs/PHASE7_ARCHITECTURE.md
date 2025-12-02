# Phase 7: Direct Spectral MC Forecasting - Architecture Design

**Date:** November 26, 2025  
**Status:** In Development  
**Expected Improvement:** 20-40% over Phase 2 (MSE: 0.0247 → 0.015-0.020)

---

## 1. Problem Statement

### Current Approach (Phase 2)
```
PACE Spectra (172 bands) 
    ↓
CNN Detector (Phase 4)
    ↓
MC Probability Maps (1 value per pixel)
    ↓
ConvLSTM Forecaster (Phase 2/6)
    ↓
Future MC Probability
```

**Issues:**
- Information loss: 172 bands → 1 value
- Error propagation: CNN errors compound in LSTM
- Two-stage training: complexity and dependency
- Limited training data: 800 map sequences

### Proposed Approach (Phase 7)
```
PACE Spectra (172 bands)
    ↓
SpectralEncoder (172 → 16 learned features)
    ↓
ConvLSTM (temporal-spatial modeling)
    ↓
Decoder (16 → MC probability)
```

**Advantages:**
- Full spectral information preserved
- Direct end-to-end learning
- Unified architecture
- 1000× more training data (1.8M pixels vs 800 sequences)

---

## 2. Data Inventory

### Available Data

**PACE Spectral Data:**
- **Granules:** 187 files (Mar 2024 - May 2025)
- **Coverage:** 440 days
- **Spectral bands:** 172 (346-719 nm)
- **Spatial coverage:** ~2,000-3,000 pixels per granule in Lake Erie bbox
- **Total pixels:** ~1.8 million (estimated)

**GLERL Ground Truth:**
- **Period:** Mar 2024 - May 2025 (overlapping with PACE)
- **Measurements:** 184 point samples
- **Unique dates:** 23 dates
- **Stations:** Multiple fixed stations in western Lake Erie
- **Binary classification:** 56 positive (≥1 µg/L) / 128 negative (30.4% / 69.6%)

### Data Characteristics

**Class Balance:**
```
MC ≥ 1.0 µg/L:  56 samples (30.4%) - POSITIVE
MC < 1.0 µg/L: 128 samples (69.6%) - NEGATIVE
```

**Spatial Coverage:**
- PACE: Full lake gridded data (84 × 73 = 6,132 pixels)
- GLERL: Point measurements at fixed stations
- **Challenge:** Very sparse labels (~184 points vs ~1.8M pixels)

---

## 3. Architecture Design

### 3.1 Model Components

#### **SpectralEncoder (Feature Extraction)**
```python
Input: (H, W, 172) - Full spectral data per pixel
    ↓
Conv2D(128, 3×3) + ReLU + Dropout(0.3)
    ↓
Conv2D(64, 3×3) + ReLU + Dropout(0.3)
    ↓
Conv2D(32, 3×3) + ReLU + Dropout(0.3)
    ↓
Conv2D(16, 1×1) + ReLU
    ↓
Output: (H, W, 16) - Learned spectral features
```

**Purpose:** 
- Compress 172 spectral bands → 16 learned features
- Discover spectral signatures for:
  - Phycocyanin (620nm) - cyanobacteria indicator
  - Chlorophyll-a (443nm, 665nm) - general algae
  - Turbidity (550nm) - water clarity
  - NIR (750nm) - water absorption

#### **ConvLSTM (Temporal-Spatial Modeling)**
```python
Input: (5, H, W, 16) - 5-day sequence of learned features
    ↓
ConvLSTM(64, 3×3, return_sequences=True)
    ↓
ConvLSTM(64, 3×3, return_sequences=True)
    ↓
ConvLSTM(32, 3×3, return_sequences=False)
    ↓
Output: (H, W, 32) - Temporal-spatial state
```

**Purpose:**
- Model temporal dynamics (bloom growth/decay)
- Preserve spatial coherence (bloom propagation)
- Output: Rich representation for final prediction

#### **Decoder (MC Probability Prediction)**
```python
Input: (H, W, 32) - ConvLSTM output
    ↓
Conv2D(32, 3×3) + ReLU + Dropout(0.2)
    ↓
Conv2D(16, 3×3) + ReLU + Dropout(0.2)
    ↓
Conv2D(8, 3×3) + ReLU + Dropout(0.2)
    ↓
Conv2D(1, 1×1) + Sigmoid
    ↓
Output: (H, W, 1) - MC probability [0,1]
```

**Purpose:**
- Map learned features → MC probability
- Binary classification: P(MC ≥ 1 µg/L)
- Sigmoid activation for probability output

### 3.2 Full Pipeline
```
Input Sequence: (5, 84, 73, 172)
    ↓ [SpectralEncoder applied to each timestep]
Features: (5, 84, 73, 16)
    ↓ [ConvLSTM temporal-spatial modeling]
State: (84, 73, 32)
    ↓ [Decoder prediction]
Output: (84, 73, 1) - MC probability map
```

**Total Parameters:** ~500k-1M (estimated)

---

## 4. Training Strategy (3 Phases)

### Phase 1: Unsupervised Pre-training (Autoencoder)

**Objective:** Learn spectral feature representations without labels

**Architecture:**
```
Encoder: (172) → (128) → (64) → (32) → (16)
Decoder: (16) → (32) → (64) → (128) → (172)
Loss: MSE(reconstructed, original)
```

**Data:**
- All PACE pixels (~1.8M)
- No labels required (self-supervised)
- Task: Reconstruct input spectra from compressed features

**Training:**
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-3
- Early stopping: patience=10

**Output:** Pre-trained SpectralEncoder weights

**Why This Works:**
- Information bottleneck (172 → 16 → 172) forces learning of essential patterns
- Features that preserve spectral information survive training
- Discovers phycocyanin peaks, chlorophyll absorption, etc. without supervision

### Phase 2: Supervised Fine-tuning (GLERL Labels)

**Objective:** Map learned features → MC classification

**Architecture:**
```
Pre-trained SpectralEncoder (frozen or fine-tuned)
    ↓
ConvLSTM (train from scratch)
    ↓
Decoder (train from scratch)
```

**Data:**
- 184 GLERL measurements (labeled)
- Binary classification: MC ≥ 1 µg/L
- Extremely sparse labels!

**Training:**
- Epochs: 100
- Batch size: 16
- Learning rate: 5e-4
- Class weighting to handle imbalance
- Data augmentation (spectral noise, spatial shifts)
- Early stopping: patience=15

**Challenge:** Only 184 labeled pixels vs 1.8M total
**Solution:** Transfer learning from Phase 1 + aggressive regularization

### Phase 3: Semi-supervised Spatial Expansion

**Objective:** Propagate labels spatially using consistency

**Strategy:**
1. Use Phase 2 model to generate pseudo-labels on unlabeled pixels
2. Filter high-confidence predictions (prob > 0.9 or < 0.1)
3. Add spatial consistency loss (nearby pixels should agree)
4. Fine-tune with mix of:
   - GLERL labels (ground truth, high weight)
   - High-confidence pseudo-labels (medium weight)
   - Spatial consistency (low weight)

**Data:**
- 184 GLERL labels
- ~900k high-confidence pseudo-labels
- Spatial consistency constraints

**Training:**
- Epochs: 50
- Batch size: 16
- Learning rate: 1e-4
- Multi-task loss (classification + consistency)

**Expected Improvement:** Better spatial coverage and smoothness

---

## 5. Loss Functions

### Phase 1 (Autoencoder)
```python
loss = MSE(reconstructed_spectra, original_spectra)
```

### Phase 2 (Supervised)
```python
# Binary cross-entropy with class weights
loss = weighted_BCE(predictions, labels, weights)

# Class weights (inverse frequency)
w_positive = n_total / (2 * n_positive)  # ~1.64
w_negative = n_total / (2 * n_negative)  # ~0.72
```

### Phase 3 (Semi-supervised)
```python
# Multi-task loss
loss = α * BCE(pred, labels)                    # Supervised
     + β * BCE(pred, pseudo_labels)             # Pseudo-labels
     + γ * spatial_consistency(pred)            # Smoothness

α = 1.0   # Ground truth weight
β = 0.3   # Pseudo-label weight
γ = 0.1   # Consistency weight
```

---

## 6. Evaluation Metrics

### Binary Classification
- **Accuracy:** Overall correctness
- **Precision:** P(true MC ≥ 1 | predicted MC ≥ 1)
- **Recall:** P(predicted MC ≥ 1 | true MC ≥ 1)
- **F1-Score:** Harmonic mean of precision/recall
- **ROC-AUC:** Area under ROC curve

### Probabilistic
- **MSE:** Mean squared error of probabilities
- **MAE:** Mean absolute error
- **Binary Cross-Entropy:** Log-loss
- **Calibration:** Reliability diagram

### Spatial
- **Coverage:** % valid predictions
- **Spatial coherence:** Moran's I (spatial autocorrelation)
- **Edge sharpness:** Gradient analysis at bloom boundaries

### Comparison to Phase 2
- **MSE:** Target < 0.020 (Phase 2: 0.0247)
- **Improvement:** > 20% required for production

---

## 7. Expected Performance

### Phase 2 Baseline
```
Test MSE: 0.0247
Test MAE: 0.1371
Coverage: 55.8%
```

### Phase 7 Targets
```
Test MSE: 0.015-0.020 (20-40% improvement)
Test MAE: 0.10-0.12
Coverage: 70-80%
Calibration: Better (direct probability learning)
```

### Success Criteria

**GO to Production:**
- MSE improvement ≥ 20% (MSE ≤ 0.020)
- Calibration improved
- Spatial coherence improved

**NO-GO:**
- MSE improvement < 5%
- Worse calibration
- Computational cost too high

**DEFER:**
- Improvement 5-20%
- Need cost-benefit analysis

---

## 8. Implementation Timeline

### Week 1: Data Preparation & Phase 1
- **Day 1-2:** Process PACE granules → spectral sequences
- **Day 3-4:** Implement and train autoencoder
- **Day 5:** Validate learned features (PCA, t-SNE)

### Week 2: Phase 2 & Phase 3
- **Day 1-2:** Implement full architecture
- **Day 3-4:** Phase 2 supervised training
- **Day 5-7:** Phase 3 semi-supervised expansion

### Week 3: Evaluation & Production
- **Day 1-2:** Comprehensive evaluation
- **Day 3-4:** Comparison analysis with Phase 2
- **Day 5:** Documentation and deployment preparation

**Total:** 2-3 weeks for complete implementation

---

## 9. Key Innovations

### vs. Phase 2 (Current)
1. **Direct spectral learning:** No CNN intermediary
2. **1000× more training data:** Unsupervised pre-training
3. **Richer features:** 172 bands vs 1 probability
4. **End-to-end:** Single unified model

### vs. Traditional Approaches
1. **Multi-phase training:** Addresses sparse labels
2. **Spatial consistency:** Leverages physical constraints
3. **Transfer learning:** Pre-training on unlabeled data
4. **Interpretable features:** Can visualize what model learns

---

## 10. Risks & Mitigation

### Risk 1: Sparse Labels (184 points)
**Mitigation:**
- Phase 1 unsupervised pre-training (no labels needed)
- Phase 3 semi-supervised expansion
- Aggressive regularization (dropout, L2)
- Data augmentation

### Risk 2: Spatial Mismatch (Point vs Grid)
**Mitigation:**
- Nearest-neighbor matching
- Spatial smoothing in loss
- Uncertainty quantification
- Validate on held-out stations

### Risk 3: Overfitting on GLERL Stations
**Mitigation:**
- Station-based cross-validation
- Test on different spatial locations
- Temporal validation (2025 test set)
- Regularization

### Risk 4: Computational Cost
**Mitigation:**
- Mixed precision training
- Efficient data pipeline (HDF5)
- Gradient checkpointing
- Model pruning if needed

---

## 11. Next Steps

### Immediate (This Session)
1. ✅ Design architecture (COMPLETE)
2. ⏳ Update data_preparation.py for binary classification
3. ⏳ Match GLERL measurements to PACE spectra
4. ⏳ Run full data preparation pipeline

### Short-term (This Week)
5. Implement SpectralEncoder + autoencoder
6. Train Phase 1 (unsupervised)
7. Validate learned features

### Medium-term (Next Week)
8. Implement full architecture
9. Train Phase 2 (supervised)
10. Train Phase 3 (semi-supervised)

### Long-term (Week 3)
11. Comprehensive evaluation
12. Comparison with Phase 2
13. Production deployment decision

---

## 12. Questions for Consideration

1. **Feature visualization:** Should we add t-SNE/PCA of learned features?
2. **Uncertainty:** Should we output prediction confidence?
3. **Multi-task:** Should we jointly predict chlorophyll + MC?
4. **Temporal horizon:** Should we extend to multi-day forecasting (2-day, 3-day)?
5. **Ensemble:** Should we combine with Phase 2 predictions?

---

**Document Status:** Living document - will update as implementation progresses
