# Phase 7: Direct Spectral Microcystin Forecasting

## Overview
Phase 7 implements direct spectral-to-toxin prediction using PACE's 172-band hyperspectral data to forecast microcystin concentrations in Lake Erie. This approach bypasses intermediate chlorophyll-a predictions and learns the direct relationship between spectral signatures and toxin levels.

## Key Innovation
Unlike Phase 2 (which predicts chlorophyll-a, then uses that to predict microcystin), Phase 7 directly maps spectral reflectance patterns to microcystin presence/absence using supervised learning from GLERL ground truth measurements.

## Architecture

### Model Components
1. **SpectralEncoder**: Compresses 172 spectral bands to 16 learned features
   - Input: (172,) raw reflectance values
   - Layers: 172 → 128 → 64 → 32 → 16
   - Activation: ReLU with batch normalization and dropout

2. **ConvLSTM**: Captures spatiotemporal patterns
   - Input: (5, 84, 73, 16) - 5 timesteps of encoded spectral maps
   - Layers: 64 → 64 → 32 filters
   - Kernel: 3×3 with recurrent dropout

3. **Decoder**: Reconstructs spatial predictions
   - Layers: 16 → 32 → 16 → 8 → 1
   - Final activation: Sigmoid (binary classification)
   - Output: (84, 73, 1) - P(MC ≥ 1.0 µg/L)

### 3-Phase Training Strategy

#### Phase 1: Unsupervised Pre-training (50 epochs)
- **Data**: ~120-150 unlabeled PACE spectral sequences
- **Objective**: Autoencoder reconstruction loss (MSE)
- **Purpose**: Learn spectral feature representations without labels
- **Loss**: Minimize ||input - reconstruction||²

#### Phase 2: Supervised Fine-tuning (100 epochs)
- **Data**: ~180-200 GLERL-labeled point measurements matched to PACE spectra
- **Objective**: Binary cross-entropy with class weights
- **Purpose**: Learn direct spectral → toxin mapping
- **Loss**: Weighted BCE (accounts for 28% positive class imbalance)

#### Phase 3: Semi-supervised Expansion (50 epochs)
- **Data**: All unlabeled sequences + high-confidence pseudo-labels
- **Objective**: Combined reconstruction + classification loss
- **Purpose**: Expand training signal using model's own predictions
- **Pseudo-labeling**: Predictions with >0.9 or <0.1 confidence

## Data

### PACE Spectral Data
- **Granules**: 187 files (March 2024 - May 2025)
- **Spectral Bands**: 172 (346-719 nm)
- **Spatial Coverage**: ~2,000-3,000 Lake Erie pixels per granule
- **Total Pixels**: ~1.8 million
- **Grid**: 84×73 cells (1.2km resolution)

### GLERL Ground Truth
- **Measurements**: 1,767 total (2012-2025)
- **PACE Overlap**: ~180-200 measurements (2024-2025)
- **Positive Rate**: 28.2% (MC ≥ 1.0 µg/L)
- **Threshold**: 1.0 µg/L (WHO recreational water guideline)

### Data Splits
- **Training**: 2024 data
- **Validation**: Through April 2025
- **Test**: May 2025 onward

## Performance Targets

### Primary Metrics
- **MSE**: 0.015-0.020 (20-40% improvement over Phase 2's 0.0247)
- **Binary Accuracy**: >85%
- **Precision**: >80% (minimize false alarms)
- **Recall**: >75% (detect most blooms)
- **AUC-ROC**: >0.90

### Success Criteria
1. MSE < 0.020 (better than Phase 2)
2. F1 score > 0.80
3. Spatially coherent predictions (no isolated pixels)
4. Temporally stable (smooth transitions day-to-day)

## Files

### Code
- `config.py` - All hyperparameters and paths
- `data_preparation.py` - PACE processing and GLERL matching
- `model.py` - Network architecture (TODO)
- `train.py` - 3-phase training pipeline (TODO)
- `utils.py` - Data loaders and utilities (TODO)
- `predict.py` - Inference and visualization (TODO)

### Data Outputs
- `glerl_pace_matched.npz` - Labeled spectral-toxin pairs (~200 samples)
- `spectral_sequences_train.h5` - Unlabeled sequences (~120 sequences)
- `spectral_sequences_val.h5` - Validation sequences
- `spectral_sequences_test.h5` - Test sequences
- `normalization_stats_train.npz` - Per-band mean/std (172 values)

### Documentation
- `../docs/PHASE7_ARCHITECTURE.md` - Comprehensive design document
- `README.md` - This file

## Usage

### 1. Data Preparation
```bash
python -m spectral_mc_forecasting.data_preparation
```

Processes all PACE granules, matches with GLERL measurements, creates temporal sequences, and saves to HDF5 files. Takes ~5-10 minutes.

### 2. Training (TODO)
```bash
# Phase 1: Unsupervised pre-training
python -m spectral_mc_forecasting.train --phase 1

# Phase 2: Supervised fine-tuning  
python -m spectral_mc_forecasting.train --phase 2

# Phase 3: Semi-supervised expansion
python -m spectral_mc_forecasting.train --phase 3
```

### 3. Prediction (TODO)
```bash
python -m spectral_mc_forecasting.predict --date 20250515
```

## Dependencies

- numpy >= 1.24
- pandas >= 2.0
- xarray >= 2023.1
- h5py >= 3.8
- scipy >= 1.10
- tensorflow >= 2.13
- tqdm >= 4.65

## Current Status

✅ **Completed**:
- Phase 7 architecture designed
- Configuration system implemented
- PACE spectral data loading
- GLERL ground truth matching
- Temporal sequence creation
- Data normalization pipeline
- Comprehensive documentation

⏳ **In Progress**:
- Data preparation pipeline running (processing 187 PACE granules)

🔜 **Next Steps**:
1. Complete data preparation run
2. Implement model architecture (model.py)
3. Implement Phase 1 training (unsupervised)
4. Implement Phase 2 training (supervised)
5. Implement Phase 3 training (semi-supervised)
6. Validation and testing
7. Prediction pipeline and visualization

## Timeline Estimate

- Data Preparation: 0.5 days ✅
- Model Implementation: 1 day
- Phase 1 Training: 0.5 days
- Phase 2 Training: 1 day
- Phase 3 Training: 0.5 days
- Evaluation & Tuning: 1 day
- Documentation: 0.5 days

**Total**: ~5 days

## Notes

- All Phase 7 code is isolated in `spectral_mc_forecasting/` directory
- Documentation follows project convention (centralized in `docs/`)
- Binary classification approach (MC ≥ 1.0 µg/L threshold)
- Spatial grid aligned with Phase 2 outputs for comparison
- Class balance is acceptable (28% positive) - using weighted loss
