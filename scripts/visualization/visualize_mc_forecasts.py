"""
Visualize MC probability forecasts vs actual during 2025 peak bloom season.

Creates side-by-side comparison plots showing:
- Input sequences (5 days of history)
- Actual MC probability map (target)
- Predicted MC probability map (forecast)
- Error map (prediction - actual)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
import os

from mc_lstm_forecasting.utils import load_mc_sequences, configure_logging
from mc_lstm_forecasting.model import build_mc_convlstm_model


def plot_forecast_comparison(X_seq, y_actual, y_pred, dates, seq_idx, save_path):
    """
    Plot comprehensive forecast comparison for a single sequence.
    
    Args:
        X_seq: Input sequence (5, 84, 73, 1)
        y_actual: Actual probability map (84, 73, 1)
        y_pred: Predicted probability map (84, 73, 1)
        dates: List of dates for the sequence
        seq_idx: Sequence index for title
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Extract 2D arrays
    actual = y_actual[:, :, 0]
    pred = y_pred[:, :, 0]
    error = pred - actual
    
    # Mask for valid (lake) pixels
    lake_mask = np.isfinite(actual)
    
    # Color map settings
    vmin, vmax = 0, 1  # Probability range
    error_max = 0.5  # Symmetric error range
    
    # --- Row 1: Input sequence (5 days) ---
    for i in range(5):
        ax = plt.subplot(4, 5, i + 1)
        input_map = X_seq[i, :, :, 0]
        
        im = ax.imshow(input_map, cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f'Input Day {i+1}\n{dates[i]}', fontsize=10)
        ax.axis('off')
        
        if i == 4:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('MC Probability', fontsize=8)
    
    # --- Row 2: Target (actual) ---
    ax_actual = plt.subplot(4, 5, 6)
    im_actual = ax_actual.imshow(actual, cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_actual.set_title(f'ACTUAL (Target)\n{dates[5]}', fontsize=11, fontweight='bold')
    ax_actual.axis('off')
    cbar = plt.colorbar(im_actual, ax=ax_actual, fraction=0.046)
    cbar.set_label('MC Probability', fontsize=8)
    
    # --- Row 2: Prediction ---
    ax_pred = plt.subplot(4, 5, 7)
    im_pred = ax_pred.imshow(pred, cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_pred.set_title(f'PREDICTED (Forecast)\n{dates[5]}', fontsize=11, fontweight='bold')
    ax_pred.axis('off')
    cbar = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046)
    cbar.set_label('MC Probability', fontsize=8)
    
    # --- Row 2: Error map ---
    ax_error = plt.subplot(4, 5, 8)
    im_error = ax_error.imshow(error, cmap='RdBu_r', vmin=-error_max, vmax=error_max, interpolation='nearest')
    ax_error.set_title('ERROR\n(Pred - Actual)', fontsize=11, fontweight='bold')
    ax_error.axis('off')
    cbar = plt.colorbar(im_error, ax=ax_error, fraction=0.046)
    cbar.set_label('Error', fontsize=8)
    
    # --- Row 3: Statistics and histograms ---
    
    # Calculate statistics on lake pixels only
    lake_actual = actual[lake_mask]
    lake_pred = pred[lake_mask]
    lake_error = error[lake_mask]
    
    mse = np.mean(lake_error ** 2)
    mae = np.mean(np.abs(lake_error))
    rmse = np.sqrt(mse)
    bias = np.mean(lake_error)
    corr = np.corrcoef(lake_actual, lake_pred)[0, 1]
    
    # Histogram of actual vs predicted
    ax_hist = plt.subplot(4, 5, 11)
    bins = np.linspace(0, 1, 30)
    ax_hist.hist(lake_actual, bins=bins, alpha=0.5, label='Actual', color='blue', density=True)
    ax_hist.hist(lake_pred, bins=bins, alpha=0.5, label='Predicted', color='red', density=True)
    ax_hist.set_xlabel('MC Probability', fontsize=9)
    ax_hist.set_ylabel('Density', fontsize=9)
    ax_hist.set_title('Distribution Comparison', fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)
    
    # Scatter plot: predicted vs actual
    ax_scatter = plt.subplot(4, 5, 12)
    ax_scatter.scatter(lake_actual, lake_pred, alpha=0.3, s=5, c='blue')
    ax_scatter.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Forecast')
    ax_scatter.set_xlabel('Actual MC Probability', fontsize=9)
    ax_scatter.set_ylabel('Predicted MC Probability', fontsize=9)
    ax_scatter.set_title(f'Correlation: {corr:.3f}', fontsize=10)
    ax_scatter.set_xlim([0, 1])
    ax_scatter.set_ylim([0, 1])
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_aspect('equal')
    
    # Error histogram
    ax_err_hist = plt.subplot(4, 5, 13)
    ax_err_hist.hist(lake_error, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax_err_hist.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax_err_hist.axvline(x=bias, color='green', linestyle='--', lw=2, label=f'Bias: {bias:.3f}')
    ax_err_hist.set_xlabel('Error (Pred - Actual)', fontsize=9)
    ax_err_hist.set_ylabel('Frequency', fontsize=9)
    ax_err_hist.set_title('Error Distribution', fontsize=10)
    ax_err_hist.legend(fontsize=8)
    ax_err_hist.grid(True, alpha=0.3)
    
    # Text statistics
    ax_stats = plt.subplot(4, 5, 14)
    ax_stats.axis('off')
    stats_text = f"""
PERFORMANCE METRICS
{'='*25}

Mean Squared Error (MSE):
  {mse:.6f}

Mean Absolute Error (MAE):
  {mae:.6f}

Root MSE (RMSE):
  {rmse:.6f}

Bias (Mean Error):
  {bias:.6f}

Correlation:
  {corr:.6f}

{'='*25}
Lake Pixels: {lake_mask.sum():,}
Total Pixels: {actual.size:,}
    """
    ax_stats.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'MC Probability Forecast - Sequence {seq_idx + 1} - Target Date: {dates[5]}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")
    print(f"    MSE={mse:.6f}, MAE={mae:.6f}, Corr={corr:.3f}")
    
    return mse, mae, rmse, bias, corr


def create_summary_plot(all_metrics, test_dates, save_path):
    """
    Create summary plot showing metrics over time.
    
    Args:
        all_metrics: List of tuples (mse, mae, rmse, bias, corr) for each sequence
        test_dates: List of target dates for each sequence
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    mse_vals = [m[0] for m in all_metrics]
    mae_vals = [m[1] for m in all_metrics]
    rmse_vals = [m[2] for m in all_metrics]
    corr_vals = [m[4] for m in all_metrics]
    
    # Parse dates
    dates = [datetime.strptime(d, '%Y%m%d') for d in test_dates]
    
    # MSE over time
    axes[0, 0].plot(dates, mse_vals, 'o-', linewidth=2, markersize=6, color='red')
    axes[0, 0].axhline(y=np.mean(mse_vals), color='blue', linestyle='--', label=f'Mean: {np.mean(mse_vals):.4f}')
    axes[0, 0].set_ylabel('MSE', fontsize=11)
    axes[0, 0].set_title('Mean Squared Error Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE over time
    axes[0, 1].plot(dates, mae_vals, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].axhline(y=np.mean(mae_vals), color='blue', linestyle='--', label=f'Mean: {np.mean(mae_vals):.4f}')
    axes[0, 1].set_ylabel('MAE', fontsize=11)
    axes[0, 1].set_title('Mean Absolute Error Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RMSE over time
    axes[1, 0].plot(dates, rmse_vals, 'o-', linewidth=2, markersize=6, color='purple')
    axes[1, 0].axhline(y=np.mean(rmse_vals), color='blue', linestyle='--', label=f'Mean: {np.mean(rmse_vals):.4f}')
    axes[1, 0].set_ylabel('RMSE', fontsize=11)
    axes[1, 0].set_title('Root Mean Squared Error Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Correlation over time
    axes[1, 1].plot(dates, corr_vals, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 1].axhline(y=np.mean(corr_vals), color='blue', linestyle='--', label=f'Mean: {np.mean(corr_vals):.4f}')
    axes[1, 1].set_ylabel('Correlation', fontsize=11)
    axes[1, 1].set_title('Correlation Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Date', fontsize=11)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    fig.suptitle('MC Probability Forecasting - Test Set Performance (2025 Aug-Oct Peak Bloom)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Summary plot saved: {save_path}")


def main():
    """Generate forecast visualizations for test set."""
    
    configure_logging()
    
    print("="*70)
    print("MC PROBABILITY FORECAST VISUALIZATION")
    print("="*70)
    
    # Create output directory
    output_dir = 'mc_lstm_forecasting/forecast_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    print("\nLoading test data...")
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
    
    # Preprocess NaN values
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test_clean = np.nan_to_num(y_test, nan=0.0)
    
    print(f"Test set: {len(X_test)} sequences (2025 Aug-Oct peak bloom)")
    
    # Build model and load weights
    print("\nLoading trained model...")
    model = build_mc_convlstm_model(input_shape=(5, 84, 73, 1))
    model.load_weights('mc_lstm_forecasting/best_model.keras')
    print("✅ Model loaded successfully!")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test, batch_size=16, verbose=1)
    
    # Get test dates
    test_dates = metadata['test_dates']
    
    print(f"\n{'='*70}")
    print("Creating forecast comparison plots...")
    print(f"{'='*70}\n")
    
    all_metrics = []
    
    # Create individual plots (show first 10 and last 5)
    indices_to_plot = list(range(min(10, len(X_test)))) + list(range(max(10, len(X_test) - 5), len(X_test)))
    
    for idx in indices_to_plot:
        # Get dates for this sequence (5 input days + 1 target day)
        target_date = test_dates[idx]
        
        # Calculate input dates (going back 5 days from target)
        # Note: actual dates may have gaps, these are approximate
        target_dt = datetime.strptime(target_date, '%Y%m%d')
        input_dates = [(target_dt - timedelta(days=6-i)).strftime('%Y%m%d') for i in range(5)]
        seq_dates = input_dates + [target_date]
        
        save_path = os.path.join(output_dir, f'forecast_{target_date}_seq{idx:02d}.png')
        
        metrics = plot_forecast_comparison(
            X_test[idx], y_test[idx], y_pred[idx],
            seq_dates, idx, save_path
        )
        all_metrics.append(metrics)
    
    # Create summary plot
    print(f"\n{'='*70}")
    print("Creating summary plot...")
    print(f"{'='*70}\n")
    
    summary_path = os.path.join(output_dir, 'test_set_summary.png')
    create_summary_plot(all_metrics, test_dates[:len(all_metrics)], summary_path)
    
    # Print overall statistics
    print(f"\n{'='*70}")
    print("OVERALL TEST SET STATISTICS")
    print(f"{'='*70}")
    
    mse_mean = np.mean([m[0] for m in all_metrics])
    mae_mean = np.mean([m[1] for m in all_metrics])
    rmse_mean = np.mean([m[2] for m in all_metrics])
    corr_mean = np.mean([m[4] for m in all_metrics])
    
    print(f"Sequences visualized: {len(all_metrics)}/{len(X_test)}")
    print(f"\nMean MSE:         {mse_mean:.6f}")
    print(f"Mean MAE:         {mae_mean:.6f}")
    print(f"Mean RMSE:        {rmse_mean:.6f}")
    print(f"Mean Correlation: {corr_mean:.6f}")
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
