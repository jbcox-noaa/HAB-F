# HAB-F Refactoring Progress

## Phase 1: Setup & Configuration ✅ COMPLETE

### Completed Tasks

1. **Directory Structure Created**
   - ✅ `chla_lstm_forecasting/` - Chlorophyll-a forecasting module
   - ✅ `microcystin_detection/` - Microcystin detection module
   - ✅ `combined_forecasting/` - Combined pipeline module
   - ✅ `visualization/` - Visualization and dashboard tools
   - ✅ `archive/` - Archived old code

2. **Old Content Archived**
   - ✅ `Notebooks/` → `archive/old_notebooks/`
   - ✅ `Scripts/` → `archive/old_scripts/`
   - ✅ `LabelData/` → `archive/label_data/`
   - ✅ `Grid_search*/` → `archive/grid_search_results/`
   - ✅ `Model_Average*/` → `archive/grid_search_results/`

3. **Configuration System Built**
   - ✅ `microcystin_detection/config.py` - Comprehensive config with all hyperparameters
   - ✅ `chla_lstm_forecasting/config.py` - LSTM model configuration
   - ✅ `combined_forecasting/config.py` - Pipeline configuration for both architectures
   - ✅ Python package structure with `__init__.py` files

4. **Documentation**
   - ✅ Updated `README.md` with complete project overview
   - ✅ Updated `.gitignore` to protect data and models
   - ✅ Updated `requirements.txt` with versioned dependencies

---

## Architecture Decision Summary

### Two Approaches to Implement

**Option 1: Spectral Forecasting First**
```
Historical PACE → CNN-LSTM → Future PACE → CNN → Future Microcystin
Spectral Data     (172 ch)   Spectral      Classifier  Risk Maps
```
- Physics-based approach
- Predicts fundamental measurements
- More interpretable
- Higher computational cost

**Option 2: Toxin Forecasting (PRIMARY)**
```
Current PACE → CNN → Microcystin → ConvLSTM → Future Microcystin
Spectral       Classifier  Maps         Temporal      Risk Maps
```
- Direct prediction of target
- More data-efficient
- Leverages temporal autocorrelation in blooms
- Easier to train and validate

**Decision**: Implement Option 2 first, then Option 1 for comparison

---

## Next Steps: Phase 2 - Refactor Microcystin Detection

### Files to Migrate from `GLERL_GT/` to `microcystin_detection/`

#### Core Files (to be refactored):
1. `utils.py` (from `helpers.py`)
   - Geographic utilities
   - Data processing functions
   - Visualization helpers
   - Retry logic
   
2. `granule_processing.py`
   - Process satellite granules
   - Extract patches around stations
   - Match temporal windows
   
3. `data_collection.py`
   - Main orchestration
   - Download satellite data
   - Process all granules
   
4. `balance_training_data.py`
   - Class balancing logic
   - Oversampling strategies
   
5. `model.py` (from `train_model.py`)
   - Extract model architecture
   - Separate from training logic
   
6. `train.py` (from `train_model.py`)
   - Training pipeline
   - Data loading
   - Model compilation
   
7. `predict.py` (from `plot_modeling.py`)
   - Generate predictions
   - Save results

#### Data Files (to be copied):
- `glrl-hab-data.csv` - Ground truth measurements
- `user-labels.csv` - User-provided labels
- `corrupted_granules.txt` - List of bad granules

#### Files to Archive (not needed):
- `application.py` - Old dashboard (will be rebuilt)
- `cnn_model_dashboard.py` - Old dashboard
- `cnn_labeler.py` - Interactive labeling tool
- `test.py` - Ad-hoc testing
- `compute_patch_stats.py` - One-off analysis
- `shap_runner.py` - Model interpretation (can revisit later)
- `oversample.py` - Merged into balance_training_data.py
- `plot_modeling copy.py` - Duplicate file
- All notebooks in `GLERL_GT/` - Archived

### Refactoring Principles

1. **Modularization**: Separate concerns (data, model, training, prediction)
2. **Configuration**: Use config.py instead of hardcoded values
3. **Clean Imports**: Explicit imports, no `from helpers import *`
4. **Type Hints**: Add type annotations for clarity
5. **Documentation**: Docstrings for all functions
6. **Error Handling**: Proper exception handling
7. **Logging**: Structured logging instead of print statements

---

## Phase 3: Refactor Chlorophyll-a Forecasting (After Phase 2)

### Files to Migrate from `CNN-LSTM/` to `chla_lstm_forecasting/`

1. `model.py` (from `LSTM.py`)
   - Extract model architecture
   - Separate parsing and data loading
   
2. `data_preparation.py` (from `LSTM.py`)
   - Data download logic
   - Sequence creation
   - Preprocessing
   
3. `train.py` (from `LSTM.py`)
   - Training pipeline
   
4. `predict.py`
   - New file for inference

### Data Experiments Needed

- [ ] Test with Sentinel-3 data (original)
- [ ] Test with PACE data (new, higher resolution)
- [ ] Compare performance

---

## Phase 4: Combined Forecasting (After Phase 3)

### New Files to Create

1. `pipeline.py`
   - Sequential inference
   - End-to-end forward pass
   
2. `train_combined.py`
   - Load pretrained models
   - Fine-tune together
   - Handle gradients
   
3. `forecast.py`
   - Multi-day forecasting
   - Autoregressive prediction
   
4. `models.py`
   - Combined architecture definitions
   - Feature fusion logic

---

## Phase 5: Visualization (After Phase 4)

### New Files to Create

1. `visualization/dashboard.py`
   - Refactor from `GLERL_GT/cnn_model_dashboard.py`
   - Support both architectures
   - Multi-day animations
   
2. `visualization/plotting.py`
   - Reusable plotting functions
   - Map generation
   - Time series plots
   
3. Demonstration Notebooks:
   - `01_data_exploration.ipynb`
   - `02_chla_forecasting_demo.ipynb`
   - `03_microcystin_detection_demo.ipynb`
   - `04_combined_forecast_demo.ipynb`

---

## Testing Strategy

### Unit Tests (to be created)
- Test data loading and preprocessing
- Test model architectures
- Test configuration validation

### Integration Tests
- End-to-end pipeline tests
- Verify data flow between modules

### Validation Tests
- Compare refactored code outputs with original
- Ensure no regressions in model performance

---

## Git Workflow

### Recommended Commit Strategy

```bash
# Phase 1 complete
git add .
git commit -m "refactor: Phase 1 - new directory structure and configuration"

# After Phase 2
git commit -m "refactor: Phase 2 - microcystin detection module"

# After Phase 3  
git commit -m "refactor: Phase 3 - chlorophyll forecasting module"

# etc.
```

### Branch Strategy (Optional)
- `main` - stable refactored code
- `refactor/phase-2` - work in progress for phase 2
- `refactor/phase-3` - work in progress for phase 3

---

## Questions & Decisions Needed

### Before Starting Phase 2:

1. **Training Data Adjustments** (you mentioned wanting to make changes)
   - What specific changes do you want to make?
   - Should we document the current process first?
   
2. **Model Validation**
   - Do you have existing trained models we should preserve?
   - Should we save the old `GLERL_GT/models/` before refactoring?
   
3. **GLERL CSV Data**
   - Is `glrl-hab-data.csv` up to date?
   - Do we need to fetch newer measurements?

---

## Current Status

**✅ COMPLETED:**
- Directory structure
- Configuration system
- Documentation
- Archive of old code

**⏳ NEXT UP:**
- Phase 2: Refactor microcystin detection module
- Copy key data files
- Extract and clean up Python modules
- Test that training still works

**Ready to proceed with Phase 2!**

---

Last Updated: 2025-11-13
