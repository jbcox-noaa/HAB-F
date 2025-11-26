"""
Training script for MC forecasting with sentinel value preprocessing.

Phase 1: Quick test using sentinel value (-1) instead of zero-fill.
Expected improvement: 10-20% over baseline.
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc_lstm_forecasting.utils import load_mc_sequences, configure_logging
from mc_lstm_forecasting.model import build_mc_convlstm_model
from mc_lstm_forecasting.preprocessing import preprocess_for_training, SENTINEL_VALUE


def main():
    """Train MC forecasting model with sentinel value preprocessing."""
    
    # Configure logging
    configure_logging()
    
    log_file = f'mc_lstm_forecasting/training_sentinel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.info("="*80)
    logging.info("PHASE 1: MC FORECASTING WITH SENTINEL VALUE")
    logging.info("="*80)
    logging.info(f"Sentinel value: {SENTINEL_VALUE}")
    logging.info("Expected improvement: 10-20% over baseline")
    
    # Step 1: Load data
    logging.info("\n" + "="*80)
    logging.info("STEP 1: LOADING DATA")
    logging.info("="*80)
    
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
    
    logging.info(f"\nData loaded:")
    logging.info(f"  Train: {len(X_train)} sequences")
    logging.info(f"  Val:   {len(X_val)} sequences")
    logging.info(f"  Test:  {len(X_test)} sequences")
    
    # Step 2: Preprocess with sentinel value
    logging.info("\n" + "="*80)
    logging.info("STEP 2: PREPROCESSING WITH SENTINEL VALUE")
    logging.info("="*80)
    
    X_train_proc, y_train_proc, train_meta = preprocess_for_training(
        X_train, y_train, method='sentinel'
    )
    X_val_proc, y_val_proc, val_meta = preprocess_for_training(
        X_val, y_val, method='sentinel'
    )
    X_test_proc, y_test_proc, test_meta = preprocess_for_training(
        X_test, y_test, method='sentinel'
    )
    
    logging.info(f"\nPreprocessing complete:")
    logging.info(f"  Train - Original NaN: {train_meta['original_nan_count_X']:,}, "
                f"Final: {train_meta['final_nan_count_X']:,}")
    logging.info(f"  Val   - Original NaN: {val_meta['original_nan_count_X']:,}, "
                f"Final: {val_meta['final_nan_count_X']:,}")
    logging.info(f"  Test  - Original NaN: {test_meta['original_nan_count_X']:,}, "
                f"Final: {test_meta['final_nan_count_X']:,}")
    
    # Verify no NaN remaining
    assert not np.any(np.isnan(X_train_proc)), "NaN values remain in X_train!"
    assert not np.any(np.isnan(y_train_proc)), "NaN values remain in y_train!"
    assert not np.any(np.isnan(X_val_proc)), "NaN values remain in X_val!"
    assert not np.any(np.isnan(y_val_proc)), "NaN values remain in y_val!"
    
    # Count sentinel values
    train_sentinel_X = (X_train_proc == SENTINEL_VALUE).sum()
    train_sentinel_y = (y_train_proc == SENTINEL_VALUE).sum()
    
    logging.info(f"\nSentinel value usage:")
    logging.info(f"  Train X: {train_sentinel_X:,} sentinel values "
                f"({100*train_sentinel_X/X_train_proc.size:.1f}%)")
    logging.info(f"  Train y: {train_sentinel_y:,} sentinel values "
                f"({100*train_sentinel_y/y_train_proc.size:.1f}%)")
    
    # Step 3: Build model
    logging.info("\n" + "="*80)
    logging.info("STEP 3: BUILDING MODEL")
    logging.info("="*80)
    
    model = build_mc_convlstm_model(
        input_shape=(5, 84, 73, 1),
        learning_rate=1e-5
    )
    
    # Step 4: Setup callbacks
    logging.info("\n" + "="*80)
    logging.info("STEP 4: SETTING UP TRAINING")
    logging.info("="*80)
    
    callbacks = [
        ModelCheckpoint(
            'mc_lstm_forecasting/best_model_sentinel.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        CSVLogger('mc_lstm_forecasting/training_history_sentinel.csv')
    ]
    
    logging.info("Callbacks configured:")
    logging.info("  - ModelCheckpoint (best_model_sentinel.keras)")
    logging.info("  - EarlyStopping (patience=10)")
    logging.info("  - ReduceLROnPlateau (factor=0.5, patience=5)")
    logging.info("  - CSVLogger (training_history_sentinel.csv)")
    
    # Step 5: Train model
    logging.info("\n" + "="*80)
    logging.info("STEP 5: TRAINING MODEL")
    logging.info("="*80)
    
    logging.info(f"\nTraining configuration:")
    logging.info(f"  Epochs: 100 (early stopping)")
    logging.info(f"  Batch size: 16")
    logging.info(f"  Learning rate: 1e-5")
    logging.info(f"  Loss function: Masked MSE (sentinel-aware)")
    logging.info(f"  Sentinel value: {SENTINEL_VALUE}")
    
    history = model.fit(
        X_train_proc, y_train_proc,
        validation_data=(X_val_proc, y_val_proc),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate on test set
    logging.info("\n" + "="*80)
    logging.info("STEP 6: EVALUATING ON TEST SET")
    logging.info("="*80)
    
    test_results = model.evaluate(X_test_proc, y_test_proc, batch_size=16, verbose=1)
    
    logging.info(f"\nTest set results:")
    logging.info(f"  Loss (MSE): {test_results[0]:.6f}")
    logging.info(f"  MAE:        {test_results[1]:.6f}")
    logging.info(f"  RMSE:       {np.sqrt(test_results[0]):.6f}")
    
    # Step 7: Compare to baseline
    logging.info("\n" + "="*80)
    logging.info("STEP 7: COMPARISON TO BASELINE")
    logging.info("="*80)
    
    baseline_mse = 0.094915
    baseline_mae = 0.254631
    
    improvement_mse = 100 * (baseline_mse - test_results[0]) / baseline_mse
    improvement_mae = 100 * (baseline_mae - test_results[1]) / baseline_mae
    
    logging.info(f"\nBaseline (zero-fill):")
    logging.info(f"  MSE: {baseline_mse:.6f}")
    logging.info(f"  MAE: {baseline_mae:.6f}")
    
    logging.info(f"\nPhase 1 (sentinel):")
    logging.info(f"  MSE: {test_results[0]:.6f}")
    logging.info(f"  MAE: {test_results[1]:.6f}")
    
    logging.info(f"\nImprovement:")
    logging.info(f"  MSE: {improvement_mse:+.1f}%")
    logging.info(f"  MAE: {improvement_mae:+.1f}%")
    
    if improvement_mse > 0:
        logging.info(f"\n✅ PHASE 1 SUCCESS: {improvement_mse:.1f}% improvement in MSE!")
    else:
        logging.info(f"\n⚠️  Phase 1 did not improve over baseline")
    
    # Save summary
    logging.info("\n" + "="*80)
    logging.info("TRAINING COMPLETE")
    logging.info("="*80)
    
    logging.info(f"\nModel saved to: mc_lstm_forecasting/best_model_sentinel.keras")
    logging.info(f"History saved to: mc_lstm_forecasting/training_history_sentinel.csv")
    logging.info(f"Log saved to: {log_file}")
    
    return history, test_results


if __name__ == '__main__':
    history, results = main()
