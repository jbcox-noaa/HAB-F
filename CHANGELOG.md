# Changelog

All notable changes to the HAB-F project will be documented in this file.

## [Unreleased] - 2024-11-26

### Repository Reorganization

**Major cleanup and restructuring before Phase 7 (Direct Spectral Forecasting) implementation.**

#### Added
- `logs/` - Centralized directory for all training and processing logs
- `tests/` - Dedicated directory for test scripts and validation tools
- `scripts/` - Organized utility scripts into subdirectories:
  - `scripts/visualization/` - Visualization and plotting scripts
  - `scripts/data_processing/` - Data download and processing utilities
  - `scripts/evaluation/` - Model evaluation and probability map generation
- `docs/archive/` - Archive for older documentation
- `visualizations/sample_outputs/` - Sample visualization outputs

#### Moved
- All `*.log` files → `logs/`
- All `training_log*.txt` files → `logs/`
- All test scripts (`test_*.py`, `validate_*.py`, `analyze_*.py`, `monitor_*.py`) → `tests/`
- All visualization scripts (`visualize_*.py`) → `scripts/visualization/`
- Data processing scripts → `scripts/data_processing/`:
  - `batch_process_pace_data.py`
  - `complete_missing_months.py`
  - `download_all_pace_data.py`
- Evaluation scripts → `scripts/evaluation/`:
  - `evaluate_mc_model.py`
  - `generate_mc_probability_maps.py`
- Shell scripts (`*.sh`) → `tests/`
- Sample PNG outputs → `visualizations/sample_outputs/`
- Documentation files → `docs/`:
  - `DATA_LEAKAGE_ANALYSIS.md`
  - `REFACTORING_PROGRESS.md`
  - `GIT_COMMIT_GUIDE.md`
  - `PHASE_2_*.md` (all Phase 2 analysis documents)
- `specimens.json` → `data/`

#### Changed
- Updated `.gitignore` to properly ignore `logs/` directory and `*.txt` files (except `requirements.txt`)
- Added explicit ignore patterns for large data directories

#### Rationale
Clean repository structure before implementing Phase 7 (Direct Spectral Forecasting), which will add:
- New spectral data preprocessing pipeline
- Spectral encoder architecture
- Multi-phase training scripts
- Additional documentation

---

## [Phase 6 Complete] - 2024-11-24

### Phase 2 Extended Training - Breakthrough Results

**MC Probability Forecasting with Dual-Channel ConvLSTM**

#### Performance
- Test MSE: **0.0247** (74.0% improvement over baseline)
- Test MAE: **0.1070**
- Best validation loss: **0.0398** (epoch 130)
- Early stopping: epoch 145 (patience=15)

#### Architecture
- Dual-channel ConvLSTM (113,697 parameters)
- Input: MC probability + validity mask (5-day history)
- Hybrid gap-filling: temporal + spatial + sentinel
- Strong regularization: L2=0.001, dropout=0.4
- Cosine annealing LR schedule (1e-3 → 1e-5 over 250 epochs)

#### Training Configuration
- Total epochs: 250 (extended from 100)
- Early stopping patience: 15
- Data augmentation: spatial flips + Gaussian noise
- Batch size: 16

#### Key Improvements Over Baseline
- Phase 1 (Sentinel): +33.2% improvement
- Phase 2 (100 epochs): +67.6% improvement
- **Phase 2 (250 epochs): +74.0% improvement** ✓ BEST

#### Documentation
- `docs/PHASE_2_BREAKTHROUGH_ANALYSIS.md` - Technical deep dive
- `docs/PHASE_2_SUCCESS_SUMMARY.md` - Executive summary
- `docs/PHASE_2_FINAL_RESULTS.md` - Complete results documentation

---

## [Phase 5 Complete] - 2024-11-20

### Baseline MC Probability Forecasting

**Initial ConvLSTM implementation for microcystin probability forecasting**

#### Performance
- Test MSE: 0.0949 (baseline)
- Simple zero-filling for missing data
- 5-day lookback, 1-day forecast

#### Issues Identified
- Zero-filling creates ambiguity (0 = missing vs 0 = low probability)
- 100% of sequences affected by missing data
- Average 41.3% missing values per sequence

---

## [Phase 4 Complete] - 2024-11-15

### Microcystin Detection Model Ensemble

**Spatial CNN for detecting microcystin presence from PACE hyperspectral data**

#### Architecture
- Dual-input CNN (patch + context features)
- Ensemble of 4 models (patch sizes: 3×3, 5×5, 7×7, 9×9)
- 172 hyperspectral bands (340-890nm)

#### Performance
- Training samples: ~5,000 from GLERL measurements
- Binary classification: MC > 0.1 µg/L
- Temporal split to prevent data leakage

#### Outputs
- Daily MC probability maps for Lake Erie
- Coverage: 84 × 73 pixel grid
- Used as input for Phase 5/6 forecasting

---

## Future Work

### Phase 7: Direct Spectral Forecasting (Planned)
- Eliminate two-stage pipeline (CNN → LSTM)
- Direct spectral → toxin prediction
- Hybrid ConvLSTM with spectral encoder
- 3-phase training: unsupervised pre-training → supervised fine-tuning → spatial expansion
- Expected: 20-40% improvement over current Phase 2 results

### Potential Improvements
- Multi-horizon forecasting (1, 3, 7 days)
- Ensemble methods for robustness
- Extended sequence length (7-day input)
- Additional features (weather, chlorophyll-a, lake levels)
- Transfer learning to other HAB species or water bodies
