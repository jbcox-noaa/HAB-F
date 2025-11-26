"""
Quick smoke test for chla_lstm_forecasting module.

Tests:
- Module imports
- Configuration access
- Model building
- Utility functions
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")
    try:
        import chla_lstm_forecasting
        from chla_lstm_forecasting import config, utils, model, train, predict
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration access."""
    print("\nTesting configuration...")
    try:
        from chla_lstm_forecasting import config
        
        assert config.SEQUENCE_LENGTH == 5
        assert config.BATCH_SIZE == 4
        assert config.LEARNING_RATE == 1e-4
        assert config.CHLA_BAND_INDEX == 21
        
        print(f"  Sequence length: {config.SEQUENCE_LENGTH}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Chlorophyll band: {config.CHLA_BAND_INDEX}")
        print("✓ Configuration accessible")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_model_building():
    """Test model building."""
    print("\nTesting model building...")
    try:
        from chla_lstm_forecasting.model import build_convlstm_model
        
        # Build a small model
        input_shape = (5, 32, 32, 2)  # (seq_len, H, W, channels)
        model = build_convlstm_model(input_shape)
        
        assert model is not None
        assert len(model.layers) > 0
        
        print(f"  Input shape: {input_shape}")
        print(f"  Model layers: {len(model.layers)}")
        print(f"  Trainable params: {model.count_params():,}")
        print("✓ Model built successfully")
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    try:
        from chla_lstm_forecasting.utils import validate_data
        
        # Create dummy data
        X = np.random.randn(10, 5, 32, 32, 2)
        y = np.random.randn(10, 32, 32, 2)
        
        # This should not raise errors
        validate_data(X, y)
        
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print("✓ Utilities working")
        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def test_public_api():
    """Test public API exposure."""
    print("\nTesting public API...")
    try:
        import chla_lstm_forecasting
        
        # Check __all__ exports
        expected = [
            'config', 'utils', 'model', 'train', 'predict',
            'parse_file', 'create_sequences', 'build_convlstm_model',
            'train_model', 'predict_single_step'
        ]
        
        for name in expected:
            assert hasattr(chla_lstm_forecasting, name), f"Missing: {name}"
        
        print(f"  Version: {chla_lstm_forecasting.__version__}")
        print(f"  Exports: {len(chla_lstm_forecasting.__all__)}")
        print("✓ Public API complete")
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("CHLA LSTM FORECASTING - SMOKE TEST")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_config,
        test_model_building,
        test_utilities,
        test_public_api
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
