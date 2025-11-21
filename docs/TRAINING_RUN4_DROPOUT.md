# Training Run 4: With Dropout Regularization
## Date: November 17, 2025

---

## Configuration

**Model Architecture Changes:**
- ✅ Added `Dropout(0.2)` after first BatchNormalization
- ✅ Added `Dropout(0.2)` after second BatchNormalization
- Total parameters: 113,697 (unchanged)

**Training Configuration:**
```
Random Seed:           42           (reproducible)
Learning Rate:         1e-4         (same as Run 1)
Batch Size:            16           (same as Run 1)
Early Stopping:        10 epochs    (more tolerant than Run 1)
Max Epochs:            100          (same as Runs 2 & 3)
Dropout Rate:          0.2          (standard regularization)
```

---

## Rationale

### Why Add Dropout?

**Problem Identified:**
- Runs 2 & 3 showed systematic overfitting
- Both failed with same pattern: epoch 1 best, then validation loss increased
- Only Run 1 succeeded (due to lucky weight initialization)
- Root cause: **No regularization** beyond BatchNormalization

**Solution:**
Dropout prevents overfitting by:
1. Randomly dropping 20% of activations during training
2. Forces network to learn redundant representations
3. Acts as ensemble of multiple sub-networks
4. Proven effective for temporal models (LSTMs, ConvLSTMs)

### Why These Hyperparameters?

**Learning Rate = 1e-4:**
- Proven in Run 1 (achieved 0.4029 test loss)
- Run 3 used 1e-5 (too slow, couldn't escape poor initialization)

**Batch Size = 16:**
- Proven in Run 1
- Run 2 used batch_size=4 (too small, noisy gradients)

**Patience = 10:**
- More tolerant than Run 1 (patience=5)
- Allows model more time to improve with dropout
- Dropout can cause more fluctuation in early epochs

**Seed = 42:**
- Ensures reproducibility
- Can re-run with same results
- Fair comparison with future experiments

---

## Expected Outcomes

### Success Criteria

✅ **Primary Goal:** Validation loss improves beyond epoch 1
- Runs 2 & 3: val_loss peaked at epoch 1, then degraded
- Run 4 (with dropout): Should show steady improvement

✅ **Secondary Goal:** Test performance matches or beats Run 1
- Run 1 benchmark: test_loss = 0.4029
- Target: test_loss ≤ 0.40

✅ **Tertiary Goal:** Reproducibility
- Same seed should give same results
- Can retrain with confidence

### Training Pattern Expected

```
Epoch 1:  val_loss ~0.51-0.52 (similar to all runs)
Epoch 2:  val_loss DECREASES (unlike Runs 2 & 3!)
Epoch 3-5: val_loss continues improving
Epoch 10-20: val_loss stabilizes or slowly improves
Best: val_loss < 0.41 (matching Run 1's ~0.413)
```

---

## Progress Log

### Epoch 1
**Status:** 🔄 IN PROGRESS

*Will be updated as training progresses...*

---

## Comparison with Previous Runs

| Run | Dropout | Seed | LR   | Patience | Best Val Loss | Test Loss | Outcome |
|-----|---------|------|------|----------|---------------|-----------|---------|
| 1   | ❌ NO   | None | 1e-4 | 5        | 0.4130 (ep 16)| **0.4029**| ✅ SUCCESS |
| 2   | ❌ NO   | None | 1e-4 | 10       | 0.5066 (ep 1) | 0.4980    | ❌ OVERFIT |
| 3   | ❌ NO   | 42   | 1e-5 | 10       | 0.5147 (ep 1) | N/A       | ❌ OVERFIT |
| 4   | ✅ YES  | 42   | 1e-4 | 10       | 🔄 TRAINING   | 🔄 TBD    | 🔄 IN PROGRESS |

---

## Architecture Comparison

### Run 1-3 (Standard Model - NO Dropout)
```
ConvLSTM2D(32) → BatchNorm → 
ConvLSTM2D(32) → BatchNorm → 
Conv2D(1)
```

### Run 4 (Standard Model - WITH Dropout)
```
ConvLSTM2D(32) → BatchNorm → Dropout(0.2) →
ConvLSTM2D(32) → BatchNorm → Dropout(0.2) →
Conv2D(1)
```

**Impact:**
- Same number of trainable parameters (113,697)
- Dropout adds zero parameters (it's just masking)
- But dramatically improves regularization

---

## Technical Details

### Model Summary

```
Model: "ConvLSTM_ChlaForecaster"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ convlstm_1 (ConvLSTM2D)         │ (None, 5, 93, 163, 32) │        39,296 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bn_1 (BatchNormalization)       │ (None, 5, 93, 163, 32) │           128 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 5, 93, 163, 32) │             0 │  ← NEW!
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ convlstm_2 (ConvLSTM2D)         │ (None, 93, 163, 32)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bn_2 (BatchNormalization)       │ (None, 93, 163, 32)    │           128 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 93, 163, 32)    │             0 │  ← NEW!
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_conv (Conv2D)            │ (None, 93, 163, 1)     │           289 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ to_float32 (Lambda)             │ (None, 93, 163, 1)     │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

Total params: 113,697 (444.13 KB)
Trainable params: 113,569 (443.63 KB)
Non-trainable params: 128 (512.00 B)
```

### Training Log Location

`training_log_dropout.txt` - Full training output with all epochs

---

## Post-Training Analysis

*This section will be updated after training completes with:*
- Final validation loss
- Test set performance
- Comparison with Run 1
- Convergence analysis
- Recommendations

---

**Status:** 🔄 **TRAINING IN PROGRESS**  
**Started:** November 17, 2025 at 11:05 AM  
**Expected Duration:** ~40-60 minutes (based on Run 1)
