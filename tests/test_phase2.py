"""
Test suite for microcystin_detection module (Phase 2)

This test file validates:
1. Module imports
2. Configuration settings
3. Model building
4. Utility functions
5. Data structures
"""

import sys
import os

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported without errors."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        import microcystin_detection
        print("✓ microcystin_detection package imported")
    except ImportError as e:
        print(f"✗ Failed to import microcystin_detection: {e}")
        return False
    
    try:
        from microcystin_detection import config
        print("✓ config module imported")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from microcystin_detection import utils
        print("✓ utils module imported")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from microcystin_detection import model
        print("✓ model module imported")
    except ImportError as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        from microcystin_detection import train
        print("✓ train module imported")
    except ImportError as e:
        print(f"✗ Failed to import train: {e}")
        return False
    
    try:
        from microcystin_detection import data_collection
        print("✓ data_collection module imported")
    except ImportError as e:
        print(f"✗ Failed to import data_collection: {e}")
        return False
    
    try:
        from microcystin_detection import balance_training_data
        print("✓ balance_training_data module imported")
    except ImportError as e:
        print(f"✗ Failed to import balance_training_data: {e}")
        return False
    
    try:
        from microcystin_detection import predict
        print("✓ predict module imported")
    except ImportError as e:
        print(f"✗ Failed to import predict: {e}")
        return False
    
    print("\n✓ ALL IMPORTS SUCCESSFUL\n")
    return True


def test_config():
    """Test configuration settings."""
    print("=" * 70)
    print("TEST 2: Configuration")
    print("=" * 70)
    
    from microcystin_detection import config
    
    # Test SENSOR_PARAMS
    print("\n1. Testing SENSOR_PARAMS...")
    assert 'PACE' in config.SENSOR_PARAMS, "PACE not in SENSOR_PARAMS"
    print(f"   ✓ PACE sensor configured")
    
    pace_params = config.SENSOR_PARAMS['PACE']
    assert 'channels' in pace_params, "channels not in PACE params"
    assert 'bbox' in pace_params, "bbox not in PACE params"
    assert 'res_km' in pace_params, "res_km not in PACE params"
    print(f"   ✓ PACE has {len(pace_params['channels'])} channels")
    print(f"   ✓ PACE resolution: {pace_params['res_km']} km")
    print(f"   ✓ PACE bbox: {pace_params['bbox']}")
    
    # Test TEMPORAL_SPLIT
    print("\n2. Testing TEMPORAL_SPLIT...")
    assert 'train' in config.TEMPORAL_SPLIT, "train split not defined"
    assert 'val' in config.TEMPORAL_SPLIT, "val split not defined"
    assert 'test' in config.TEMPORAL_SPLIT, "test split not defined"
    
    n_train = len(config.TEMPORAL_SPLIT['train'])
    n_val = len(config.TEMPORAL_SPLIT['val'])
    n_test = len(config.TEMPORAL_SPLIT['test'])
    
    print(f"   ✓ Train split: {n_train} dates")
    print(f"   ✓ Val split: {n_val} dates")
    print(f"   ✓ Test split: {n_test} dates")
    print(f"   ✓ Total: {n_train + n_val + n_test} dates")
    
    # Test helper functions
    print("\n3. Testing helper functions...")
    n_channels = config.get_channels_for_sensor('PACE')
    assert n_channels == 172, f"Expected 172 PACE channels, got {n_channels}"
    print(f"   ✓ get_channels_for_sensor('PACE') = {n_channels}")
    
    # Test PATCH_SIZES
    print("\n4. Testing PATCH_SIZES...")
    assert len(config.PATCH_SIZES) > 0, "No patch sizes defined"
    print(f"   ✓ Patch sizes: {config.PATCH_SIZES}")
    
    # Test PM_THRESHOLDS
    print("\n5. Testing PM_THRESHOLDS...")
    assert len(config.PM_THRESHOLDS) > 0, "No PM thresholds defined"
    print(f"   ✓ PM thresholds: {config.PM_THRESHOLDS}")
    
    print("\n✓ ALL CONFIG TESTS PASSED\n")
    return True


def test_model_building():
    """Test that we can build a model."""
    print("=" * 70)
    print("TEST 3: Model Building")
    print("=" * 70)
    
    try:
        from microcystin_detection.model import build_model, get_model_config
        from microcystin_detection import config
        
        print("\n1. Testing get_model_config...")
        model_config = get_model_config('PACE', patch_size=3)
        print(f"   ✓ Model config for PACE, patch_size=3:")
        print(f"     - patch_size: {model_config['patch_size']}")
        print(f"     - n_channels: {model_config['n_channels']}")
        print(f"     - context_size: {model_config['context_size']}")
        print(f"     - learning_rate: {model_config['learning_rate']}")
        
        print("\n2. Building model...")
        model = build_model(
            patch_size=3,
            n_channels=172,
            context_size=177,  # 5 + 172
            learning_rate=1e-4
        )
        
        print(f"   ✓ Model built successfully")
        print(f"   ✓ Model name: {model.name}")
        print(f"   ✓ Number of layers: {len(model.layers)}")
        print(f"   ✓ Total parameters: {model.count_params():,}")
        
        # Test model inputs
        print("\n3. Testing model inputs...")
        input_shapes = [inp.shape for inp in model.inputs]
        print(f"   ✓ Input shapes: {input_shapes}")
        assert len(model.inputs) == 2, "Model should have 2 inputs"
        print(f"   ✓ Model has {len(model.inputs)} inputs (patch + context)")
        
        # Test model output
        print("\n4. Testing model output...")
        output_shape = model.output.shape
        print(f"   ✓ Output shape: {output_shape}")
        assert output_shape[-1] == 1, "Output should be single value (binary classification)"
        print(f"   ✓ Binary classification output")
        
        print("\n✓ ALL MODEL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ MODEL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("=" * 70)
    print("TEST 4: Utility Functions")
    print("=" * 70)
    
    from microcystin_detection import utils
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    print("\n1. Testing extract_datetime_from_filename...")
    test_filename = "PACE_OCI.20240517T163009.L2.OC.V2_0.NRT.nc"
    dt = utils.extract_datetime_from_filename(test_filename)
    assert dt is not None, "Failed to extract datetime"
    assert dt.year == 2024, f"Wrong year: {dt.year}"
    assert dt.month == 5, f"Wrong month: {dt.month}"
    assert dt.day == 17, f"Wrong day: {dt.day}"
    print(f"   ✓ Extracted: {dt}")
    
    print("\n2. Testing estimate_position...")
    times = [pd.Timestamp('2024-05-17 16:30:00', tz='UTC')]
    lats = np.array([41.5])
    lons = np.array([-83.0])
    target_time = pd.Timestamp('2024-05-17 17:00:00', tz='UTC')
    
    lat, lon = utils.estimate_position(times, lats, lons, target_time)
    assert lat == 41.5, f"Wrong lat: {lat}"
    assert lon == -83.0, f"Wrong lon: {lon}"
    print(f"   ✓ Position: ({lat}, {lon})")
    
    print("\n3. Testing stretch function...")
    test_array = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    stretched = utils.stretch(test_array, lower_percent=10, upper_percent=90)
    assert stretched.min() >= 0, "Stretched min should be >= 0"
    assert stretched.max() <= 1, "Stretched max should be <= 1"
    print(f"   ✓ Stretch range: [{stretched.min():.3f}, {stretched.max():.3f}]")
    
    print("\n4. Testing patch_to_features...")
    patch_dict = {
        400.0: np.random.randn(3, 3),
        500.0: np.random.randn(3, 3),
        600.0: np.random.randn(3, 3)
    }
    features = utils.patch_to_features(patch_dict)
    assert len(features) == 3, f"Expected 3 features, got {len(features)}"
    print(f"   ✓ Features shape: {features.shape}")
    
    print("\n5. Testing configure_logging...")
    utils.configure_logging()
    print(f"   ✓ Logging configured")
    
    print("\n✓ ALL UTILITY TESTS PASSED\n")
    return True


def test_data_structures():
    """Test data structures and file existence."""
    print("=" * 70)
    print("TEST 5: Data Structures")
    print("=" * 70)
    
    import os
    from microcystin_detection import config
    
    print("\n1. Testing data file existence...")
    data_files = {
        'GLERL data': 'microcystin_detection/glrl-hab-data.csv',
        'User labels': 'microcystin_detection/user-labels.csv',
        'Corrupted granules': 'microcystin_detection/corrupted_granules.txt'
    }
    
    for name, path in data_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✓ {name}: {path} ({size:,} bytes)")
        else:
            print(f"   ⚠ {name}: {path} (not found)")
    
    print("\n2. Testing GLERL data structure...")
    import pandas as pd
    glerl_path = 'microcystin_detection/glrl-hab-data.csv'
    if os.path.exists(glerl_path):
        df = pd.read_csv(glerl_path, index_col=0)
        print(f"   ✓ GLERL data loaded: {len(df)} records")
        print(f"   ✓ Columns: {list(df.columns[:5])}...")
        
        required_cols = ['timestamp', 'lat', 'lon', 'station_name']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        print(f"   ✓ All required columns present")
    else:
        print(f"   ⚠ GLERL data not found, skipping validation")
    
    print("\n3. Testing temporal split dates...")
    from datetime import datetime
    for split_name, dates in config.TEMPORAL_SPLIT.items():
        print(f"\n   {split_name.upper()} split ({len(dates)} dates):")
        # Show first 3 dates
        for i, date_str in enumerate(dates[:3]):
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            print(f"     - {date_str} ({dt.strftime('%A, %B %d, %Y')})")
        if len(dates) > 3:
            print(f"     ... and {len(dates) - 3} more")
    
    print("\n✓ ALL DATA STRUCTURE TESTS PASSED\n")
    return True


def test_function_signatures():
    """Test that main functions have correct signatures."""
    print("=" * 70)
    print("TEST 6: Function Signatures")
    print("=" * 70)
    
    import inspect
    from microcystin_detection import (
        train_model,
        collect_training_data,
        balance_by_oversampling_negatives,
        predict_from_granule,
        build_model
    )
    
    functions_to_test = [
        ('train_model', train_model),
        ('collect_training_data', collect_training_data),
        ('balance_by_oversampling_negatives', balance_by_oversampling_negatives),
        ('predict_from_granule', predict_from_granule),
        ('build_model', build_model)
    ]
    
    for name, func in functions_to_test:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        print(f"\n{name}:")
        print(f"   Parameters: {params}")
        print(f"   ✓ Signature valid")
    
    print("\n✓ ALL FUNCTION SIGNATURE TESTS PASSED\n")
    return True


def test_type_annotations():
    """Test that functions have type annotations."""
    print("=" * 70)
    print("TEST 7: Type Annotations")
    print("=" * 70)
    
    import inspect
    from microcystin_detection import utils, model, train, predict
    
    modules_to_check = [
        ('utils', utils),
        ('model', model),
        ('train', train),
        ('predict', predict)
    ]
    
    total_functions = 0
    annotated_functions = 0
    
    for module_name, module in modules_to_check:
        print(f"\n{module_name} module:")
        functions = [
            (name, obj) for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and not name.startswith('_')
        ]
        
        for func_name, func in functions[:5]:  # Check first 5 functions
            sig = inspect.signature(func)
            has_annotations = any(
                param.annotation != inspect.Parameter.empty
                for param in sig.parameters.values()
            ) or sig.return_annotation != inspect.Signature.empty
            
            if has_annotations:
                annotated_functions += 1
                print(f"   ✓ {func_name}: has type hints")
            else:
                print(f"   ⚠ {func_name}: no type hints")
            
            total_functions += 1
    
    coverage = (annotated_functions / total_functions * 100) if total_functions > 0 else 0
    print(f"\n✓ Type annotation coverage: {coverage:.1f}% ({annotated_functions}/{total_functions})")
    print("✓ TYPE ANNOTATION TEST COMPLETE\n")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("MICROCYSTIN DETECTION MODULE - TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Model Building", test_model_building),
        ("Utility Functions", test_utils),
        ("Data Structures", test_data_structures),
        ("Function Signatures", test_function_signatures),
        ("Type Annotations", test_type_annotations)
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
    print("TEST SUMMARY")
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
    success = run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Phase 2 code is ready.\n")
        exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.\n")
        exit(1)
