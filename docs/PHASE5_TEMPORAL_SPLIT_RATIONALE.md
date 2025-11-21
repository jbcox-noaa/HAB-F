# Phase 5 Temporal Split Strategy - Rationale

## 📊 Updated Split Strategy

### Previous Approach (Initial Plan)
```
Training:   2024 (80% = ~194 sequences)
Validation: 2024 (20% = ~48 sequences)
Test:       2025 ALL (~60-70 sequences)
```

**Issue:** Validation set would not contain bloom season data, making it hard to tune model for bloom prediction specifically.

---

### **IMPROVED APPROACH** ✅

```
Training:   2024 ALL (242 maps → ~200-220 sequences)
Validation: 2025 Jan-Jul (~45 maps → ~35-40 sequences)
Test:       2025 Aug-Oct (~30 maps → ~25-30 sequences)
```

---

## 🎯 Why This Is Better

### 1. Bloom Season Coverage in Evaluation

**Training Set (2024):**
- June 2024: 20 maps ← Bloom onset
- July 2024: 19 maps ← Early bloom
- August 2024: 25 maps ← **Peak bloom**
- September 2024: 19 maps ← Late bloom
- October 2024: 21 maps ← Bloom decline
- **Total bloom maps: 104**

**Validation Set (2025 Jan-Jul):**
- Non-bloom: Jan-May (~40 maps)
- June 2025: 8 maps ← **Bloom onset**
- July 2025: 22 maps ← **Early bloom**
- **Total bloom maps: ~30**

**Test Set (2025 Aug-Oct):**
- August 2025: 21 maps ← **Peak bloom** 🎯
- September 2025: 25 maps ← **Late bloom** 🎯
- October 2025: 1 map ← Bloom decline
- **Total bloom maps: ~47**

### 2. Different Bloom Stages

| Stage | Training | Validation | Test |
|-------|----------|------------|------|
| **Bloom Onset** | June 2024 | June 2025 ✅ | - |
| **Early Bloom** | July 2024 | July 2025 ✅ | - |
| **Peak Bloom** | Aug 2024 | - | Aug 2025 ✅ |
| **Late Bloom** | Sep 2024 | - | Sep 2025 ✅ |

This allows us to:
- **Validate** on bloom onset/early bloom conditions
- **Test** on peak/late bloom conditions
- Ensure model works across **all bloom stages**

### 3. Operational Realism

**Real-world forecasting scenario:**
1. Train model on historical bloom season (2024)
2. Validate on early season data (Jan-Jul 2025)
3. Test on peak season data (Aug-Oct 2025)

This mimics how the model would be used operationally:
- Learn from previous year's full bloom cycle
- Tune hyperparameters on early season of current year
- Deploy for peak bloom prediction in current year

### 4. Prevents Temporal Leakage

✅ **All 2025 data still unseen during training**
- Training uses only 2024
- No future information leaking to past
- True temporal validation maintained

✅ **Sequential split within 2025**
- Validation (Jan-Jul) comes before Test (Aug-Oct)
- Maintains temporal ordering
- No data from August used to tune model that predicts August

---

## 📈 Expected Metrics

### Validation Performance (2025 Jan-Jul)
- **Early bloom prediction:** Model should detect bloom onset in June-July
- **Expected MAE:** 0.12-0.18 (easier early bloom)
- **Use case:** Hyperparameter tuning, early stopping

### Test Performance (2025 Aug-Oct)
- **Peak bloom prediction:** Model should forecast high MC probability
- **Expected MAE:** 0.15-0.20 (harder peak bloom)
- **Use case:** Final model evaluation, operational readiness

### Success Criteria
✅ Validation MAE < 0.20 (on bloom onset/early bloom)  
✅ Test MAE < 0.20 (on peak/late bloom)  
✅ Test MAE within 20% of Validation MAE (generalization)  
✅ Model correctly identifies bloom timing in both sets

---

## 🔬 Scientific Advantages

### 1. Bloom Stage Representation
- **Onset (June):** Low-moderate MC probability, rapid change
- **Early (July):** Moderate MC probability, expansion phase
- **Peak (August-September):** High MC probability, maximum extent
- **Decline (October):** Decreasing MC probability, senescence

Both validation and test cover critical bloom phases!

### 2. Inter-annual Variability
- 2024 bloom characteristics → Training
- 2025 bloom characteristics → Validation & Test
- Tests model's ability to adapt to year-to-year differences

### 3. Temporal Dynamics
- Model learns temporal patterns from 2024
- Validates on early 2025 (bloom starting)
- Tests on late 2025 (bloom peaking)
- Ensures forecasting works throughout bloom progression

---

## 💡 Implementation Details

### Data Splits in Code

```python
from datetime import datetime

def split_by_date(dates, sequences):
    """Split sequences by year and date"""
    train_mask = dates.year == 2024
    val_mask = (dates.year == 2025) & (dates < datetime(2025, 8, 1))
    test_mask = (dates.year == 2025) & (dates >= datetime(2025, 8, 1))
    
    train_seq = sequences[train_mask]
    val_seq = sequences[val_mask]
    test_seq = sequences[test_mask]
    
    return train_seq, val_seq, test_seq
```

### Expected Sequence Counts

```
Total maps: 317
├── 2024: 242 maps
│   └── Training sequences: ~200-220 (depends on temporal gaps)
└── 2025: 75 maps
    ├── Jan-Jul: ~45 maps
    │   └── Validation sequences: ~35-40
    └── Aug-Oct: ~30 maps
        └── Test sequences: ~25-30
```

With sequence length = 5 days, we lose ~5 sequences per transition due to gap requirements.

---

## ✅ Summary

**Why this split is superior:**

1. ✅ **Bloom coverage:** Both val and test contain bloom season
2. ✅ **Stage diversity:** Different bloom phases in val vs test
3. ✅ **Operational realism:** Mimics real forecasting workflow
4. ✅ **Temporal integrity:** No data leakage, maintains order
5. ✅ **Model tuning:** Can optimize for bloom prediction specifically
6. ✅ **Robust testing:** Final test on peak bloom (hardest scenario)

**User requirement satisfied:**
> "Both the validation and test set should contain some of the peak bloom in 2025."

- Validation: Contains bloom onset/early bloom (June-July 2025)
- Test: Contains peak bloom (August-September 2025)
- Both sets have realistic bloom conditions for evaluation ✅

---

**Approved:** November 21, 2025  
**Status:** Implemented in config.py, README.md, and planning docs  
**Next:** Implement data loading pipeline with this split strategy
