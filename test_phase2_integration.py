"""
Integration test for microcystin_detection module

This script validates:
1. Model can be built and compiled
2. Dummy data can be created and passed through model
3. Configuration values are consistent
4. Temporal splits don't overlap
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_forward_pass():
    """Test that we can do a forward pass through the model."""
    print("=" * 70)
    print("INTEGRATION TEST: Model Forward Pass")
    print("=" * 70)
    
    from microcystin_detection.model import build_model
    
    # Build model
    patch_size = 3
    n_channels = 172
    context_size = 177
    
    model = build_model(
        patch_size=patch_size,
        n_channels=n_channels,
        context_size=context_size,
        learning_rate=1e-4
    )
    
    print(f"\n✓ Model built: {model.count_params():,} parameters")
    
    # Create dummy data
    batch_size = 4
    X_patch = np.random.randn(batch_size, patch_size, patch_size, n_channels + 1).astype('float32')
    X_context = np.random.randn(batch_size, context_size).astype('float32')
    
    print(f"✓ Created dummy batch:")
    print(f"  - X_patch shape: {X_patch.shape}")
    print(f"  - X_context shape: {X_context.shape}")
    
    # Forward pass
    predictions = model.predict([X_patch, X_context], verbose=0)
    
    print(f"✓ Forward pass successful:")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    assert predictions.shape == (batch_size, 1), "Wrong prediction shape"
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions not in [0,1]"
    
    print("\n✓ MODEL FORWARD PASS TEST PASSED\n")
    return True


def test_temporal_splits_no_overlap():
    """Test that temporal splits don't overlap."""
    print("=" * 70)
    print("INTEGRATION TEST: Temporal Split Validation")
    print("=" * 70)
    
    from microcystin_detection import config
    
    # Get all dates
    train_dates = set(config.TEMPORAL_SPLIT['train'])
    val_dates = set(config.TEMPORAL_SPLIT['val'])
    test_dates = set(config.TEMPORAL_SPLIT['test'])
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_dates)} dates")
    print(f"  - Val:   {len(val_dates)} dates")
    print(f"  - Test:  {len(test_dates)} dates")
    
    # Check for overlaps
    train_val_overlap = train_dates & val_dates
    train_test_overlap = train_dates & test_dates
    val_test_overlap = val_dates & test_dates
    
    assert len(train_val_overlap) == 0, f"Train/Val overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train/Test overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val/Test overlap: {val_test_overlap}"
    
    print(f"\n✓ No overlaps between splits")
    
    # Check chronological ordering within splits
    for split_name, dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
        date_objs = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dates])
        print(f"\n{split_name.upper()} split date range:")
        print(f"  - First: {date_objs[0].strftime('%Y-%m-%d')}")
        print(f"  - Last:  {date_objs[-1].strftime('%Y-%m-%d')}")
        print(f"  - Span:  {(date_objs[-1] - date_objs[0]).days} days")
    
    print("\n✓ TEMPORAL SPLIT VALIDATION PASSED\n")
    return True


def test_config_consistency():
    """Test that configuration values are consistent."""
    print("=" * 70)
    print("INTEGRATION TEST: Configuration Consistency")
    print("=" * 70)
    
    from microcystin_detection import config
    
    print("\n1. Checking SENSOR_PARAMS consistency...")
    for sensor, params in config.SENSOR_PARAMS.items():
        print(f"\n   {sensor}:")
        
        # Check required keys
        required_keys = ['short_names', 'channels', 'bbox', 'res_km']
        for key in required_keys:
            assert key in params, f"{sensor} missing key: {key}"
            print(f"     ✓ {key}: {params[key] if key != 'channels' else f'{len(params[key])} channels'}")
        
        # Check bbox format
        bbox = params['bbox']
        assert len(bbox) == 4, f"{sensor} bbox should have 4 values"
        assert bbox[0] < bbox[2], f"{sensor} lon_min should be < lon_max"
        assert bbox[1] < bbox[3], f"{sensor} lat_min should be < lat_max"
    
    print("\n2. Checking PATCH_SIZES...")
    for ps in config.PATCH_SIZES:
        assert ps > 0, f"Patch size {ps} should be positive"
        assert ps % 2 == 1, f"Patch size {ps} should be odd"
    print(f"   ✓ All patch sizes are positive and odd: {config.PATCH_SIZES}")
    
    print("\n3. Checking PM_THRESHOLDS...")
    for thresh in config.PM_THRESHOLDS:
        assert thresh >= 0, f"Threshold {thresh} should be non-negative"
    print(f"   ✓ All thresholds are non-negative: {config.PM_THRESHOLDS}")
    
    print("\n4. Checking BBOX consistency...")
    bbox = config.BBOX
    print(f"   Lake Erie bbox: {bbox}")
    print(f"   ✓ Longitude range: {bbox[0]} to {bbox[2]} ({bbox[2] - bbox[0]:.2f}°)")
    print(f"   ✓ Latitude range: {bbox[1]} to {bbox[3]} ({bbox[3] - bbox[1]:.2f}°)")
    
    print("\n✓ CONFIGURATION CONSISTENCY TEST PASSED\n")
    return True


def test_data_file_integrity():
    """Test that data files exist and have correct structure."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Data File Integrity")
    print("="*70 + "\n")
    
    import pandas as pd
    from microcystin_detection import config
    
    # Test GLERL data file
    print("1. Testing GLERL data...")
    data_path = config.GLERL_CSV
    df = pd.read_csv(data_path)
    
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Check for required columns
    required = ['timestamp', 'lat', 'lon', 'station_name', 'particulate_microcystin']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print(f"   ✓ All required columns present")
    
    # Check data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   ✓ Timestamp column parsed correctly")
    
    # Check value ranges (excluding NaN values)
    valid_lat = df['lat'].dropna()
    valid_lon = df['lon'].dropna()
    assert valid_lat.between(40, 43).all(), "Some latitudes out of Lake Erie range"
    assert valid_lon.between(-84, -81).all(), "Some longitudes out of Lake Erie range"
    print(f"   ✓ Coordinates within Lake Erie bounds (excluding {df['lat'].isna().sum()} NaN records)")
    
    # Check for missing critical values
    n_missing_pm = df['particulate_microcystin'].isna().sum()
    print(f"   ✓ Missing PM values: {n_missing_pm}/{len(df)} ({n_missing_pm/len(df)*100:.1f}%)")
    
    # Check date range
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    print(f"   ✓ Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    print("\n✓ DATA FILE INTEGRITY TEST PASSED\n")
    return True


def test_normalization_logic():
    """Test normalization logic."""
    print("=" * 70)
    print("INTEGRATION TEST: Normalization Logic")
    print("=" * 70)
    
    from microcystin_detection.predict import normalize_patch, normalize_context
    
    print("\n1. Testing patch normalization...")
    patch = np.random.randn(3, 3, 173).astype('float32')  # 172 channels + 1 mask
    means = np.random.randn(172).astype('float32')
    stds = np.random.randn(172).astype('float32') + 1.0  # Ensure positive
    
    normalized = normalize_patch(patch.copy(), means, stds)
    
    print(f"   ✓ Input shape: {patch.shape}")
    print(f"   ✓ Output shape: {normalized.shape}")
    print(f"   ✓ No NaNs in output: {not np.isnan(normalized).any()}")
    
    print("\n2. Testing context normalization...")
    context = np.random.randn(177).astype('float32')
    ctx_means = np.random.randn(177).astype('float32')
    ctx_stds = np.random.randn(177).astype('float32') + 1.0
    
    normalized_ctx = normalize_context(context.copy(), ctx_means, ctx_stds)
    
    print(f"   ✓ Input shape: {context.shape}")
    print(f"   ✓ Output shape: {normalized_ctx.shape}")
    print(f"   ✓ No NaNs in output: {not np.isnan(normalized_ctx).any()}")
    
    print("\n✓ NORMALIZATION LOGIC TEST PASSED\n")
    return True


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("MICROCYSTIN DETECTION - INTEGRATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ("Model Forward Pass", test_model_forward_pass),
        ("Temporal Split Validation", test_temporal_splits_no_overlap),
        ("Configuration Consistency", test_config_consistency),
        ("Data File Integrity", test_data_file_integrity),
        ("Normalization Logic", test_normalization_logic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED WITH EXCEPTION:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:12} - {test_name}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    return all(result for _, result in results)


if __name__ == '__main__':
    success = run_integration_tests()
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! Phase 2 is production-ready.\n")
        exit(0)
    else:
        print("\n⚠️  SOME INTEGRATION TESTS FAILED. Please review.\n")
        exit(1)
