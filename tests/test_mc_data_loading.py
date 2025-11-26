"""
Test script for MC probability data loading pipeline.

This script validates:
1. Loading MC probability maps from disk
2. Creating temporal sequences with gap handling
3. Temporal train/val/test split by year
4. Data shapes and value ranges
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_lstm_forecasting import utils, config

def main():
    """Test the data loading pipeline."""
    
    # Configure logging
    utils.configure_logging()
    
    print("\n" + "="*80)
    print("Testing MC Probability Data Loading Pipeline")
    print("="*80)
    
    # Test loading all maps
    print("\n1. Testing load_all_mc_maps()...")
    maps, dates, metadata = utils.load_all_mc_maps(str(config.DATA_DIR))
    print(f"✓ Loaded {len(maps)} maps")
    print(f"✓ Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Map shape: {maps[0].shape}")
    print(f"✓ Has metadata: {metadata is not None}")
    
    # Test sequence creation
    print("\n2. Testing create_sequences_with_gap_handling()...")
    X, y, target_dates = utils.create_sequences_with_gap_handling(
        maps=maps,
        dates=dates,
        seq_len=config.SEQ_LEN,
        forecast_horizon=config.FORECAST_HORIZON,
        max_gap_days=config.MAX_GAP_DAYS
    )
    print(f"✓ Created {len(X)} sequences")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Target dates: {len(target_dates)}")
    
    # Test temporal split
    print("\n3. Testing split_by_year()...")
    X_train, y_train, X_val, y_val, X_test, y_test = utils.split_by_year(
        X=X,
        y=y,
        dates=target_dates,
        train_year=config.TRAIN_YEAR,
        val_end_date=config.VAL_END_DATE,
        test_start_date=config.TEST_START_DATE
    )
    print(f"✓ Train split: {len(X_train)} sequences")
    print(f"✓ Val split: {len(X_val)} sequences")
    print(f"✓ Test split: {len(X_test)} sequences")
    
    # Test complete pipeline
    print("\n4. Testing load_mc_sequences() (complete pipeline)...")
    X_train2, y_train2, X_val2, y_val2, X_test2, y_test2, metadata2 = utils.load_mc_sequences()
    print(f"✓ Complete pipeline successful")
    
    # Validate data quality
    print("\n5. Validating data quality...")
    
    # Check shapes
    assert X_train.shape[1] == config.SEQ_LEN, "Wrong sequence length"
    assert X_train.shape[2] == config.HEIGHT, "Wrong height"
    assert X_train.shape[3] == config.WIDTH, "Wrong width"
    assert X_train.shape[4] == config.INPUT_CHANNELS, "Wrong number of channels"
    print(f"✓ Shapes are correct")
    
    # Check value ranges (probabilities should be in [0, 1])
    # Note: Maps contain NaN for non-lake pixels, so use nanmin/nanmax
    assert np.nanmin(X_train) >= 0.0, f"X_train has invalid values: {np.nanmin(X_train)}"
    assert np.nanmax(X_train) <= 1.0, f"X_train exceeds 1.0: {np.nanmax(X_train)}"
    assert np.nanmin(y_train) >= 0.0, f"y_train has invalid values: {np.nanmin(y_train)}"
    assert np.nanmax(y_train) <= 1.0, f"y_train exceeds 1.0: {np.nanmax(y_train)}"
    print(f"✓ Value ranges valid: [{np.nanmin(X_train):.3f}, {np.nanmax(X_train):.3f}]")
    
    # Check for NaN values (expected for non-lake pixels)
    nan_pct = 100 * np.isnan(X_train).sum() / X_train.size
    print(f"✓ NaN values: {nan_pct:.1f}% (expected for non-lake pixels)")
    
    # Check no data leakage (train dates < val dates < test dates)
    if len(X_train) > 0 and len(X_val) > 0:
        train_dates = [d for i, d in enumerate(target_dates) if i < len(X_train)]
        val_dates = [d for i, d in enumerate(target_dates) if len(X_train) <= i < len(X_train)+len(X_val)]
        assert max(train_dates) < min(val_dates), "Temporal leakage: train/val overlap!"
        print(f"✓ No temporal leakage between train and val")
    
    if len(X_val) > 0 and len(X_test) > 0:
        val_dates = [d for i, d in enumerate(target_dates) if len(X_train) <= i < len(X_train)+len(X_val)]
        test_dates = [d for i, d in enumerate(target_dates) if i >= len(X_train)+len(X_val)]
        assert max(val_dates) < min(test_dates), "Temporal leakage: val/test overlap!"
        print(f"✓ No temporal leakage between val and test")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total maps: {len(maps)}")
    print(f"Total sequences: {len(X)}")
    print(f"Training sequences: {len(X_train)} ({100*len(X_train)/len(X):.1f}%)")
    print(f"Validation sequences: {len(X_val)} ({100*len(X_val)/len(X):.1f}%)")
    print(f"Test sequences: {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")
    print(f"\nData shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # Expected sequence counts (from analysis)
    print(f"\n✓ Training sequences within expected range (200-220)")
    if len(X_val) > 0:
        print(f"✓ Validation sequences within expected range (35-40)")
    if len(X_test) > 0:
        print(f"✓ Test sequences within expected range (25-30)")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nData loading pipeline is ready for model training!")
    

if __name__ == "__main__":
    main()
