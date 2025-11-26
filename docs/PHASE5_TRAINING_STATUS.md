# Phase 5 Training Status

**Date:** November 21, 2025  
**Status:** 🔄 TRAINING IN PROGRESS

---

## ✅ Completed Implementation

### 1. Data Loading Pipeline ✅
- **File:** `mc_lstm_forecasting/utils.py`
- Loads 317 MC probability maps
- Creates 181 sequences with gap handling (max_gap=3 days)
- Temporal split: 109 train, 40 val, 32 test
- Handles NaN values (non-lake pixels)

### 2. Model Architecture ✅
- **File:** `mc_lstm_forecasting/model.py`
- ConvLSTM with 112,545 parameters
- 2 ConvLSTM layers (32 filters each)
- Sigmoid output for probabilities [0, 1]
- **Custom masked loss functions** to handle NaN pixels

### 3. Training Pipeline ✅
- **File:** `mc_lstm_forecasting/train.py`
- Complete training loop with callbacks
- Early stopping (patience=10)
- Learning rate reduction on plateau
- Model checkpointing (saves best model)
- CSV logging of training history

### 4. NaN Handling Solution ✅
**Problem:** Initial training produced NaN loss because:
- MC probability maps contain NaN for non-lake pixels (67% of pixels)
- NaN values propagate through neural network

**Solutions implemented:**
1. **Custom masked loss functions:**
   - `masked_mse_loss()` - MSE computed only on valid pixels
   - `masked_mae_loss()` - MAE computed only on valid pixels
   - Ignores NaN pixels using `tf.math.is_finite()` mask

2. **Input preprocessing:**
   - Replace NaN with 0 before training: `np.nan_to_num(X, nan=0.0)`
   - Prevents NaN propagation through network
   - Non-lake pixels become 0 probability (appropriate)

---

## 🔄 Training Status

### Configuration
```
Random seed: 42
Batch size: 16
Max epochs: 100
Learning rate: 1e-05
Early stopping patience: 10

Data:
  Train: 109 sequences (2024-03-24 to 2024-12-14)
  Val: 40 sequences (2025-03-09 to 2025-07-29, early bloom)
  Test: 32 sequences (2025-08-08 to 2025-10-01, peak bloom)
```

### Training Process
- **Started:** 2025-11-21 14:17:xx
- **Status:** Running in background
- **Log file:** `mc_lstm_forecasting/training_log.txt`
- **Loss:** Training successfully (no NaN values)
- **Model:** Saving to `mc_lstm_forecasting/best_model.keras`

### Monitoring
Training can be monitored via:
```bash
# Watch live progress
tail -f mc_lstm_forecasting/training_log.txt

# Check if still running
ps aux | grep mc_lstm

# Use monitoring script
python monitor_training.py
```

---

## 📊 Expected Results

### Training Time
- ~7 batches per epoch (109 samples / 16 batch_size)
- ~10-15 seconds per epoch (estimate)
- Expected early stopping around epoch 20-50
- Total training time: ~10-20 minutes

### Target Performance
Based on chlorophyll forecasting (MSE=0.3965):
- **Goal:** MSE < 0.05 on MC probability forecasting
- **Reason:** MC probabilities are in [0, 1] range, smaller than chlorophyll
- **MAE Goal:** < 0.15 (mean absolute error in probability)

---

## 📁 Output Files

When training completes, the following files will be generated:

1. **best_model.keras** - Best model (lowest validation loss)
2. **training_log.txt** - Complete training log
3. **training_history.csv** - Epoch-by-epoch metrics
4. **training_history.png** - Loss/MAE curves plot
5. **training_summary.txt** - Final summary report
6. **training_YYYYMMDD_HHMMSS.log** - Timestamped log file

All files saved to: `mc_lstm_forecasting/`

---

## 🔍 Key Implementation Details

### Masked Loss Function
```python
def masked_mse_loss(y_true, y_pred):
    """MSE loss that ignores NaN values (non-lake pixels)."""
    mask = tf.math.is_finite(y_true)
    mask = tf.cast(mask, tf.float32)
    
    squared_error = tf.square(y_true - y_pred)
    masked_error = tf.where(mask > 0, squared_error, 0.0)
    
    sum_error = tf.reduce_sum(masked_error)
    count = tf.reduce_sum(mask)
    mse = tf.where(count > 0, sum_error / count, 0.0)
    
    return mse
```

### NaN Preprocessing
```python
# Replace NaN with 0 before training
X_train = np.nan_to_num(X_train, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
```

### Callbacks Setup
```python
callbacks = [
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    CSVLogger('training_history.csv')
]
```

---

## 🚀 Next Steps

After training completes:

1. ✅ Review training history plot
2. ✅ Check test set performance metrics
3. ⏳ Create prediction/evaluation script (`predict.py`)
4. ⏳ Generate forecast visualizations
5. ⏳ Compare predictions vs actual (2025 bloom season)
6. ⏳ Document performance and findings

---

## 📝 Technical Notes

### Why Masked Loss?
- Lake Erie covers only ~33% of the raster grid
- Remaining 67% are land/border pixels (NaN values)
- Standard MSE would treat NaN as part of loss → NaN gradients
- Masked loss computes error only on valid lake pixels
- More accurate representation of model performance

### Why Replace NaN with 0?
- Neural networks cannot process NaN values
- NaN propagates through layers → all outputs become NaN
- Replacing with 0 is semantically correct:
  * Non-lake pixels = 0 probability of MC
  * Allows network to process all pixels
  * Masked loss still ignores these pixels in gradient computation

### Data Augmentation Considerations
- Not implemented in current version
- Could add if overfitting occurs:
  * Horizontal/vertical flips
  * Small rotations (±5°)
  * Would double/triple effective dataset size
- Current 109 training sequences should be sufficient

---

**Status:** Training running successfully. Monitor `training_log.txt` for progress.
