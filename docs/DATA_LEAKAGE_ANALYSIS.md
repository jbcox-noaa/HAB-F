# Data Leakage Analysis: Microcystin Detection Model

## Executive Summary

**The 100% test accuracy is likely unrealistic and caused by multiple severe data leakage issues.**

---

## Critical Data Leakage Issues Identified

### 🚨 **ISSUE #1: Data Augmentation BEFORE Train/Test Split** 
**Severity: CRITICAL**

**What's happening:**
```python
# Current sequence in train.py (lines 318-327):
X_patch, X_context, y_binary = augment_data(...)  # Line 318
splits = create_train_val_test_split(X_patch, X_context, y_binary)  # Line 327
```

**The problem:**
- Augmentation creates 4 versions of each sample (original + 3 flips)
- THEN the split randomly distributes these 4 versions across train/val/test
- Result: **Original image in train, its flipped version in test**
- These are nearly identical → model has already "seen" test samples

**Impact:** Massive performance inflation. The model is essentially memorizing slightly different versions of training samples.

**Evidence:**
- 1,545 total samples → After augmentation: 6,180 samples
- Random split means ~4,635 train, ~618 val, ~927 test
- High probability that augmented versions of the same original are split across sets

---

### 🚨 **ISSUE #2: Random Temporal Split (Not Date-Based)**
**Severity: CRITICAL**

**What's happening:**
```python
# train_test_split uses random_state=42, stratify=y
# Does NOT consider dates or temporal ordering
```

**The problem:**
- 83 unique dates in dataset
- 75 dates have multiple samples (up to 46 samples per date!)
- Random split puts samples from SAME DAY in train AND test sets
- Example: Date 2024-09-01 has 20 samples → ~15 train, ~2 val, ~3 test

**Temporal distribution:**
- **Positive samples:** July-October 2024 (95 day range)
- **Negative samples:** May 2024 - April 2025 (354 day range)

**Impact:** Model learns temporal/seasonal patterns from test period dates that also appear in training.

**Why this matters:**
- Lake conditions on same day are highly correlated
- Water quality doesn't change much hour-to-hour
- Model can learn "if date is Sept 2024 → probably positive"

---

### ⚠️ **ISSUE #3: Same-Station, Same-Day Duplicates**
**Severity: MODERATE**

**Findings:**
- 212 date-station combinations with multiple samples
- Example: 2024-05-08 at WE2: 4 samples (all negative)
- Same location, same day → nearly identical environmental conditions

**The problem:**
- Multiple patch sizes extracted from same granule at same station
- Patches from patch_3, patch_5, patch_7, patch_9 are spatially overlapping
- Random split can put overlapping patches in different sets

**Impact:** Spatial autocorrelation leakage.

---

### ⚠️ **ISSUE #4: Station-Specific Patterns**
**Severity: MODERATE**

**Findings:**
- 707 unique stations total
- 8 stations appear in BOTH positive and negative samples
- Model can learn station-specific characteristics

**Examples of stations in both classes:**
- These 8 stations provide temporal evolution data
- Model might learn "WE2 in summer → positive, WE2 in winter → negative"

**Impact:** Model learns location shortcuts rather than spectral signatures.

---

### ⚠️ **ISSUE #5: Highly Imbalanced Temporal Coverage**
**Severity: LOW-MODERATE**

**Distribution:**
```
Positive samples by month:
  2024-09: 167 samples (32%)
  2024-07: 136 samples (26%)
  2024-08: 119 samples (23%)
  2024-10: 93 samples (18%)

Negative samples by month:
  2025-03: 406 samples (39%)  ← Winter oversampling!
  2024-12: 199 samples (19%)
  2024-06: 104 samples (10%)
```

**The problem:**
- Positive samples: concentrated in summer (July-Sept)
- Negative samples: concentrated in winter (Dec-Mar)
- Model can learn seasonal shortcuts

**Impact:** Model learns "summer months → bloom risk" rather than spectral features.

---

## Evidence of Overfitting

### Training Progression Analysis:
```
Epoch 1:   val_loss = 0.6444, val_acc = 67.5%
Epoch 100: val_loss = 0.0412, val_acc = 100%  ← Perfect accuracy reached
Epoch 200: val_loss = 0.0033, val_acc = 100%  ← Loss still decreasing
Epoch 300: val_loss = 0.0005, val_acc = 100%  ← Final
```

**Observations:**
1. Validation accuracy hits 100% around epoch 100
2. Model continues training for 200 more epochs
3. Loss keeps decreasing but accuracy can't improve (already perfect)
4. **This is textbook overfitting to the validation set**

### Test Results:
```
Test Accuracy:  100.0%
Test Precision: 100.0%
Test Recall:    100.0%
Test F1 Score:  100.0%

Confusion Matrix:
  TN=407, FP=0
  FN=0,   TP=69
```

**Why this is suspicious:**
- Real-world classification tasks rarely achieve 100% on held-out test data
- Especially with environmental/remote sensing data (noise, clouds, etc.)
- Perfect metrics strongly suggest the test set was "seen" during training

---

## Root Cause Summary

| Issue | Mechanism | Severity |
|-------|-----------|----------|
| Augmentation before split | Flipped versions leak to test | **CRITICAL** |
| Random temporal split | Same-day samples in train+test | **CRITICAL** |
| Overlapping spatial patches | Same location patches split | **MODERATE** |
| Station memorization | Location-based shortcuts | **MODERATE** |
| Seasonal concentration | Temporal shortcuts | **LOW-MODERATE** |

**Combined effect:** Model has multiple "backdoors" to achieve high accuracy without learning true spectral signatures of microcystin.

---

## Recommended Fixes

### 1. **Fix Augmentation Leakage** (CRITICAL)
```python
# WRONG (current):
X_patch, X_context, y_binary = augment_data(...)
splits = create_train_val_test_split(X_patch, X_context, y_binary)

# CORRECT:
# Split FIRST, then augment only training set
splits = create_train_val_test_split(X_patch, X_context, y_binary)
X_patch_train, X_context_train, y_train = augment_data(
    X_patch_train, X_context_train, y_train
)
# Do NOT augment val/test sets
```

### 2. **Fix Temporal Leakage** (CRITICAL)
```python
# Extract dates from samples
dates = [extract_date(sample) for sample in raw_data]

# Split by date, not randomly
# Option A: Date-based cutoff
train_dates = dates < '2024-09-01'
val_dates = (dates >= '2024-09-01') & (dates < '2024-10-01')
test_dates = dates >= '2024-10-01'

# Option B: GroupKFold by date
from sklearn.model_selection import GroupKFold
groups = [date.strftime('%Y%m%d') for date in dates]
gkf = GroupKFold(n_splits=5)
```

### 3. **Fix Spatial Leakage** (MODERATE)
```python
# Split by unique (date, station) combinations
# Ensure all samples from same date-station go to same split
sample_ids = [f"{date}_{station}" for date, station in zip(dates, stations)]
groups = np.array(sample_ids)

# Use GroupKFold or GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
```

### 4. **Retrain with Proper Splits**
- Implement fixes 1-3
- Re-train all 4 models
- **Expect accuracy to drop to 80-90%** (more realistic)
- If still >95%, investigate further

---

## Expected Outcomes After Fixes

### Realistic Performance Estimates:
- **Accuracy:** 80-90% (down from 100%)
- **F1 Score:** 0.75-0.85 (down from 1.0)
- **Confusion Matrix:** Will have false positives and false negatives

### Why lower is actually better:
- Demonstrates model generalizes to truly unseen data
- More trustworthy for deployment
- Honest assessment of capabilities
- Identifies areas for improvement

---

## Action Items

1. ✅ **STOP current training** - Don't train patch_5, patch_7, patch_9 yet
2. ⚠️ **Fix train.py** - Implement temporal split and post-split augmentation
3. ⚠️ **Re-collect or re-split data** - Use date-based groups
4. ⚠️ **Re-train patch_3** - Validate fixes work
5. ⚠️ **Compare results** - Document performance change
6. ⚠️ **Train remaining models** - Only after confirming fixes

---

## Questions to Consider

1. **Do we have enough data for temporal splits?**
   - 83 dates total
   - Need ~60 train, ~12 val, ~12 test dates
   - May need to collect more historical data

2. **Should we use k-fold cross-validation?**
   - Given limited dates, GroupKFold might be better
   - Provides multiple train/test splits
   - More robust performance estimates

3. **What's the true baseline?**
   - Class distribution: 33% positive, 67% negative
   - Random classifier: 50% accuracy
   - Always-predict-negative: 67% accuracy
   - **Our model should significantly beat 67%**

---

## Conclusion

The 100% test accuracy is almost certainly caused by data leakage, primarily from:
1. Augmenting before splitting (augmented versions leak to test)
2. Random temporal splits (same-day samples in both train and test)

**This doesn't mean the model is bad** - it means we need to fix the evaluation pipeline to get honest metrics. The spectral features are likely informative, but the current setup doesn't properly test generalization.

**Recommended immediate action:** Fix the training pipeline before proceeding with ensemble training.
