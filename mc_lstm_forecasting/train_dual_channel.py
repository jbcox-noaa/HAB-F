"""
Training script for MC forecasting with dual-channel architecture.

Phase 2: Dual-channel model with hybrid gap-filling and explicit validity mask.
Expected improvement: Additional 15-20% over Phase 1 sentinel approach.
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
from mc_lstm_forecasting.model import build_mc_convlstm_dual_channel
from mc_lstm_forecasting.preprocessing import create_dual_channel_input, SENTINEL_VALUE


def main():
    """Train dual-channel MC forecasting model with hybrid gap-filling."""
    
    # Configure logging
    configure_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'mc_lstm_forecasting/training_dual_channel_{timestamp}.log'
    
    logging.info("="*80)
    logging.info("PHASE 2: DUAL-CHANNEL MC FORECASTING")
    logging.info("="*80)
    logging.info("Architecture: 2 channels (probability + validity mask)")
    logging.info("Gap-filling: Hybrid (temporal + spatial)")
    logging.info(f"Sentinel value: {SENTINEL_VALUE}")
    logging.info("Expected improvement: 15-20% over Phase 1")
    
    # Step 1: Load data
    logging.info("\n" + "="*80)
    logging.info("STEP 1: LOADING DATA")
    logging.info("="*80)
    
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
    
    logging.info(f"\nData loaded:")
    logging.info(f"  Train: {len(X_train)} sequences")
    logging.info(f"  Val:   {len(X_val)} sequences")
    logging.info(f"  Test:  {len(X_test)} sequences")
    logging.info(f"  Original shape: {X_train.shape}")
    
    # Step 2: Create dual-channel input with hybrid gap-filling
    logging.info("\n" + "="*80)
    logging.info("STEP 2: CREATING DUAL-CHANNEL INPUT")
    logging.info("="*80)
    
    X_train_dual, y_train_proc, train_meta = create_dual_channel_input(
        X_train, y_train, 
        gap_fill_method='hybrid',
        temporal_max_days=3,
        spatial_max_dist=3
    )
    
    X_val_dual, y_val_proc, val_meta = create_dual_channel_input(
        X_val, y_val, 
        gap_fill_method='hybrid',
        temporal_max_days=3,
        spatial_max_dist=3
    )
    
    X_test_dual, y_test_proc, test_meta = create_dual_channel_input(
        X_test, y_test, 
        gap_fill_method='hybrid',
        temporal_max_days=3,
        spatial_max_dist=3
    )
    
    logging.info("\nPreprocessed data:")
    logging.info(f"  Train shape: {X_train_dual.shape}")
    logging.info(f"  Val shape:   {X_val_dual.shape}")
    logging.info(f"  Test shape:  {X_test_dual.shape}")
    logging.info(f"  Channels: 0=Probability (gap-filled), 1=Validity mask")
    
    # Log coverage statistics
    logging.info("\nTrain set coverage:")
    logging.info(f"  Original data: {train_meta['coverage_stats']['original_pct']:.1f}%")
    logging.info(f"  Temporal fill: {train_meta['coverage_stats']['temporal_pct']:.1f}%")
    logging.info(f"  Spatial fill:  {train_meta['coverage_stats']['spatial_pct']:.1f}%")
    logging.info(f"  Unfilled:      {train_meta['coverage_stats']['unfilled_pct']:.1f}%")
    
    # Step 3: Build dual-channel model
    logging.info("\n" + "="*80)
    logging.info("STEP 3: BUILDING DUAL-CHANNEL MODEL (BALANCED REGULARIZATION)")
    logging.info("="*80)
    logging.info("ANTI-OVERFITTING MEASURES (BALANCED):")
    logging.info("  - Dropout: 0.3 (moderate, not too aggressive)")
    logging.info("  - L2 regularization: 5e-5 (lighter weight penalty)")
    logging.info("  - Batch size: 32 (stable gradients)")
    logging.info("  - Learning rate: 7e-4 (balanced exploration/exploitation)")
    
    model = build_mc_convlstm_dual_channel(
        input_shape=(5, 84, 73, 2),  # 2 channels!
        filters_1=32,
        filters_2=32,
        kernel_size=(3, 3),
        learning_rate=7e-4,  # BALANCED: Between 5e-4 (worked) and 1e-3 (too high)
        loss='masked_mse',
        sentinel=SENTINEL_VALUE,
        dropout=0.3,  # BALANCED: Between 0.2 (too weak) and 0.4 (too strong)
        l2_reg=5e-5   # BALANCED: Lighter than 1e-4 which was too restrictive
    )
    
    model.summary(print_fn=logging.info)
    
    # Step 4: Setup callbacks
    logging.info("\n" + "="*80)
    logging.info("STEP 4: CONFIGURING TRAINING")
    logging.info("="*80)
    
    callbacks = [
        ModelCheckpoint(
            'mc_lstm_forecasting/best_model_dual_channel.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # REDUCED: More aggressive LR reduction
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            f'mc_lstm_forecasting/training_history_dual_channel_{timestamp}.csv'
        )
    ]
    
    logging.info("Callbacks configured:")
    logging.info("  - ModelCheckpoint (save best)")
    logging.info("  - EarlyStopping (patience=10)")
    logging.info("  - ReduceLROnPlateau (factor=0.5, patience=3) - AGGRESSIVE")
    logging.info("  - CSVLogger")
    
    # Step 5: Train model
    logging.info("\n" + "="*80)
    logging.info("STEP 5: TRAINING (IMPROVED REGULARIZATION)")
    logging.info("="*80)
    
    history = model.fit(
        X_train_dual, y_train_proc,
        validation_data=(X_val_dual, y_val_proc),
        epochs=50,
        batch_size=32,  # INCREASED: Better gradient estimates
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate on test set
    logging.info("\n" + "="*80)
    logging.info("STEP 6: EVALUATION ON TEST SET")
    logging.info("="*80)
    
    test_loss, test_mae = model.evaluate(X_test_dual, y_test_proc, verbose=0)
    
    logging.info(f"\nTest Results:")
    logging.info(f"  Test Loss (MSE): {test_loss:.6f}")
    logging.info(f"  Test MAE:        {test_mae:.6f}")
    logging.info(f"  Test RMSE:       {np.sqrt(test_loss):.6f}")
    
    # Step 7: Compare to baseline and Phase 1
    logging.info("\n" + "="*80)
    logging.info("STEP 7: COMPARISON TO PREVIOUS PHASES")
    logging.info("="*80)
    
    # These are the known results from previous runs
    baseline_mse = 0.094915
    baseline_mae = 0.254631
    phase1_mse = 0.063398
    phase1_mae = 0.219647
    
    phase2_mse = test_loss
    phase2_mae = test_mae
    
    improvement_over_baseline = ((baseline_mse - phase2_mse) / baseline_mse) * 100
    improvement_over_phase1 = ((phase1_mse - phase2_mse) / phase1_mse) * 100
    
    logging.info("Performance Comparison:")
    logging.info(f"\nBaseline (zero-fill):")
    logging.info(f"  MSE: {baseline_mse:.6f}")
    logging.info(f"  MAE: {baseline_mae:.6f}")
    
    logging.info(f"\nPhase 1 (sentinel value):")
    logging.info(f"  MSE: {phase1_mse:.6f}")
    logging.info(f"  MAE: {phase1_mae:.6f}")
    logging.info(f"  Improvement over baseline: {((baseline_mse - phase1_mse) / baseline_mse * 100):.1f}%")
    
    logging.info(f"\nPhase 2 (dual-channel + hybrid fill):")
    logging.info(f"  MSE: {phase2_mse:.6f}")
    logging.info(f"  MAE: {phase2_mae:.6f}")
    logging.info(f"  Improvement over baseline: {improvement_over_baseline:.1f}%")
    logging.info(f"  Improvement over Phase 1: {improvement_over_phase1:.1f}%")
    
    # Success check
    if phase2_mse < phase1_mse:
        logging.info(f"\n{'='*80}")
        logging.info("✅ PHASE 2 SUCCESS! Dual-channel model outperforms Phase 1!")
        logging.info(f"{'='*80}")
    else:
        logging.info(f"\n{'='*80}")
        logging.info("⚠️  Phase 2 did not improve over Phase 1")
        logging.info(f"{'='*80}")
    
    if phase2_mse < 0.05:
        logging.info("🎯 TARGET ACHIEVED: MSE < 0.05!")
    
    # Save final results
    results = {
        'phase2_mse': float(phase2_mse),
        'phase2_mae': float(phase2_mae),
        'phase2_rmse': float(np.sqrt(phase2_mse)),
        'improvement_over_baseline': float(improvement_over_baseline),
        'improvement_over_phase1': float(improvement_over_phase1),
        'training_epochs': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'metadata': {
            'train_coverage': train_meta['coverage_stats'],
            'test_coverage': test_meta['coverage_stats']
        }
    }
    
    import json
    with open('mc_lstm_forecasting/results_dual_channel.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to: mc_lstm_forecasting/results_dual_channel.json")
    logging.info(f"Model saved to: mc_lstm_forecasting/best_model_dual_channel.keras")
    
    return model, history, results


if __name__ == '__main__':
    model, history, results = main()
