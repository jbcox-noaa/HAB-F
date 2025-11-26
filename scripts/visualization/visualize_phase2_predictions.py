"""
Visualize Phase 2 Improved model predictions vs ground truth.

This script generates comprehensive visualizations of model outputs:
1. Side-by-side prediction vs ground truth maps
2. Difference/error maps
3. Temporal sequences showing 5-day input + prediction
4. Scatter plots of predicted vs actual values
5. Spatial error distribution analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from pathlib import Path
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mc_lstm_forecasting.utils import load_mc_sequences
from mc_lstm_forecasting.preprocessing import create_dual_channel_input, SENTINEL_VALUE
from mc_lstm_forecasting.model import CastToFloat32, masked_mse_loss, masked_mae_loss


def create_mc_colormap():
    """Create a colormap for microcystin probability."""
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mc_risk', colors, N=n_bins)
    return cmap


def plot_prediction_comparison(X_seq, y_true, y_pred, idx, save_path):
    """
    Plot a single prediction: 5 input frames + prediction + ground truth + error.
    
    Args:
        X_seq: (5, 84, 73, 2) - Input sequence with dual channels
        y_true: (84, 73, 1) - Ground truth
        y_pred: (84, 73, 1) - Prediction
        idx: Sequence index
        save_path: Where to save the figure
    """
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
    
    cmap = create_mc_colormap()
    
    # Plot 5 input frames (channel 0 = probability)
    for t in range(5):
        ax = fig.add_subplot(gs[0, t])
        prob_map = X_seq[t, :, :, 0]  # Probability channel
        validity_map = X_seq[t, :, :, 1]  # Validity channel
        
        # Mask sentinel values
        prob_display = np.where(prob_map == SENTINEL_VALUE, np.nan, prob_map)
        
        im = ax.imshow(prob_display, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Input Day {t+1}\n(Validity: {(validity_map > 0).mean():.1%})', 
                     fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Add validity indicator
        avg_validity = validity_map.mean()
        validity_color = 'green' if avg_validity > 0.7 else 'orange' if avg_validity > 0.4 else 'red'
        ax.text(0.02, 0.98, f'Valid: {avg_validity:.1%}', 
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=validity_color, alpha=0.6),
                verticalalignment='top', color='white')
    
    # Add colorbar for input frames
    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('MC Probability', fontsize=11, fontweight='bold')
    
    # Plot ground truth
    ax_true = fig.add_subplot(gs[1, 1])
    y_true_display = np.where(y_true[:, :, 0] == SENTINEL_VALUE, np.nan, y_true[:, :, 0])
    im_true = ax_true.imshow(y_true_display, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax_true.set_title('Ground Truth\n(Day 6)', fontsize=12, fontweight='bold')
    ax_true.axis('off')
    
    # Plot prediction
    ax_pred = fig.add_subplot(gs[1, 2])
    y_pred_display = np.where(y_true[:, :, 0] == SENTINEL_VALUE, np.nan, y_pred[:, :, 0])
    im_pred = ax_pred.imshow(y_pred_display, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax_pred.set_title('Prediction\n(Day 6)', fontsize=12, fontweight='bold')
    ax_pred.axis('off')
    
    # Plot absolute error
    ax_error = fig.add_subplot(gs[1, 3])
    error = np.abs(y_true[:, :, 0] - y_pred[:, :, 0])
    error_display = np.where(y_true[:, :, 0] == SENTINEL_VALUE, np.nan, error)
    im_error = ax_error.imshow(error_display, cmap='Reds', vmin=0, vmax=0.5, aspect='auto')
    ax_error.set_title('Absolute Error\n|Truth - Pred|', fontsize=12, fontweight='bold')
    ax_error.axis('off')
    
    # Add error colorbar
    cbar_error = fig.colorbar(im_error, ax=ax_error)
    cbar_error.set_label('Error', fontsize=10, fontweight='bold')
    
    # Plot error statistics
    ax_stats = fig.add_subplot(gs[1, 4])
    ax_stats.axis('off')
    
    # Calculate metrics (excluding sentinel)
    valid_mask = y_true[:, :, 0] != SENTINEL_VALUE
    if valid_mask.sum() > 0:
        mse = np.mean(error[valid_mask]**2)
        mae = np.mean(error[valid_mask])
        rmse = np.sqrt(mse)
        max_error = np.max(error[valid_mask])
        
        # Calculate by risk level
        high_risk_mask = valid_mask & (y_true[:, :, 0] > 0.7)
        med_risk_mask = valid_mask & (y_true[:, :, 0] > 0.4) & (y_true[:, :, 0] <= 0.7)
        low_risk_mask = valid_mask & (y_true[:, :, 0] <= 0.4)
        
        stats_text = "PREDICTION METRICS\n" + "="*30 + "\n\n"
        stats_text += f"Overall:\n"
        stats_text += f"  MSE:  {mse:.6f}\n"
        stats_text += f"  MAE:  {mae:.6f}\n"
        stats_text += f"  RMSE: {rmse:.6f}\n"
        stats_text += f"  Max Error: {max_error:.6f}\n\n"
        
        stats_text += f"By Risk Level:\n"
        if high_risk_mask.sum() > 0:
            stats_text += f"  High (>0.7): MAE={np.mean(error[high_risk_mask]):.4f}\n"
        if med_risk_mask.sum() > 0:
            stats_text += f"  Med (0.4-0.7): MAE={np.mean(error[med_risk_mask]):.4f}\n"
        if low_risk_mask.sum() > 0:
            stats_text += f"  Low (<0.4): MAE={np.mean(error[low_risk_mask]):.4f}\n"
        
        stats_text += f"\nValid Pixels: {valid_mask.sum():,}\n"
        stats_text += f"Coverage: {valid_mask.mean():.1%}"
    else:
        stats_text = "No valid pixels\nin this prediction"
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle(f'Sequence {idx}: 5-Day Input → 1-Day Forecast', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scatter_analysis(y_true_all, y_pred_all, save_path):
    """Create scatter plot of predicted vs actual values."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Flatten and filter sentinel values
    y_true_flat = y_true_all.flatten()
    y_pred_flat = y_pred_all.flatten()
    valid_mask = y_true_flat != SENTINEL_VALUE
    
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    # 1. Overall scatter
    ax1 = axes[0, 0]
    ax1.hexbin(y_true_valid, y_pred_valid, gridsize=50, cmap='YlOrRd', mincnt=1)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Ground Truth MC Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted MC Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted vs Ground Truth\n(Hexbin Density)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add R² and metrics
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true_valid, y_pred_valid)
    r_squared = r**2
    mse = np.mean((y_true_valid - y_pred_valid)**2)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    
    ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}\nMSE = {mse:.6f}\nMAE = {mae:.6f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = y_pred_valid - y_true_valid
    ax2.hexbin(y_true_valid, residuals, gridsize=50, cmap='RdBu_r', mincnt=1)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax2.set_xlabel('Ground Truth MC Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Pred - Truth)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis\n(Hexbin Density)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # 3. Error distribution
    ax3 = axes[1, 0]
    abs_errors = np.abs(residuals)
    ax3.hist(abs_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.4f}')
    ax3.axvline(np.median(abs_errors), color='orange', linestyle='--', linewidth=2, 
                label=f'Median = {np.median(abs_errors):.4f}')
    ax3.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Error by risk level
    ax4 = axes[1, 1]
    
    # Categorize by risk level
    low_mask = y_true_valid < 0.4
    med_mask = (y_true_valid >= 0.4) & (y_true_valid < 0.7)
    high_mask = y_true_valid >= 0.7
    
    risk_levels = ['Low\n(<0.4)', 'Medium\n(0.4-0.7)', 'High\n(≥0.7)']
    mae_by_risk = [
        np.mean(abs_errors[low_mask]) if low_mask.sum() > 0 else 0,
        np.mean(abs_errors[med_mask]) if med_mask.sum() > 0 else 0,
        np.mean(abs_errors[high_mask]) if high_mask.sum() > 0 else 0
    ]
    counts = [low_mask.sum(), med_mask.sum(), high_mask.sum()]
    
    colors = ['#2c7bb6', '#fdae61', '#d7191c']
    bars = ax4.bar(risk_levels, mae_by_risk, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add counts on bars
    for bar, count, mae_val in zip(bars, counts, mae_by_risk):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'MAE: {mae_val:.4f}\nn={count:,}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_title('MAE by Risk Level', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, max(mae_by_risk) * 1.3 if max(mae_by_risk) > 0 else 0.1)
    
    fig.suptitle('Phase 2 Improved: Prediction Quality Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spatial_error_analysis(y_true_all, y_pred_all, save_path):
    """Analyze spatial distribution of errors."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Calculate mean error map
    valid_mask = y_true_all[:, :, :, 0] != SENTINEL_VALUE
    
    # Mean absolute error at each pixel
    abs_errors = np.abs(y_true_all[:, :, :, 0] - y_pred_all[:, :, :, 0])
    mean_abs_error = np.where(valid_mask.sum(axis=0) > 0,
                              np.sum(abs_errors * valid_mask, axis=0) / np.maximum(valid_mask.sum(axis=0), 1),
                              np.nan)
    
    # Mean ground truth at each pixel
    mean_true = np.where(valid_mask.sum(axis=0) > 0,
                         np.sum(y_true_all[:, :, :, 0] * valid_mask, axis=0) / np.maximum(valid_mask.sum(axis=0), 1),
                         np.nan)
    
    # Mean prediction at each pixel
    mean_pred = np.where(valid_mask.sum(axis=0) > 0,
                         np.sum(y_pred_all[:, :, :, 0] * valid_mask, axis=0) / np.maximum(valid_mask.sum(axis=0), 1),
                         np.nan)
    
    # Bias (systematic over/under prediction)
    bias = mean_pred - mean_true
    
    # 1. Mean Absolute Error Map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(mean_abs_error, cmap='Reds', vmin=0, vmax=0.3, aspect='auto')
    ax1.set_title('Mean Absolute Error\n(Spatial Distribution)', fontsize=13, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('MAE', fontsize=11, fontweight='bold')
    
    # 2. Mean Ground Truth
    ax2 = axes[0, 1]
    im2 = ax2.imshow(mean_true, cmap=create_mc_colormap(), vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Mean Ground Truth\n(Average MC Probability)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
    cbar2.set_label('MC Prob', fontsize=11, fontweight='bold')
    
    # 3. Mean Prediction
    ax3 = axes[1, 0]
    im3 = ax3.imshow(mean_pred, cmap=create_mc_colormap(), vmin=0, vmax=1, aspect='auto')
    ax3.set_title('Mean Prediction\n(Average Forecast)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('MC Prob', fontsize=11, fontweight='bold')
    
    # 4. Bias Map
    ax4 = axes[1, 1]
    im4 = ax4.imshow(bias, cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='auto')
    ax4.set_title('Prediction Bias\n(Pred - Truth, Red=Over, Blue=Under)', 
                  fontsize=13, fontweight='bold')
    ax4.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046)
    cbar4.set_label('Bias', fontsize=11, fontweight='bold')
    
    fig.suptitle('Spatial Error Analysis: Where Does the Model Struggle?', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Generate comprehensive prediction visualizations."""
    
    print("="*80)
    print("PHASE 2 IMPROVED: PREDICTION VISUALIZATION")
    print("="*80)
    
    # Load model
    print("\n1. Loading trained model...")
    model_path = 'mc_lstm_forecasting/best_model_dual_channel_v2.keras'
    
    # Load with custom objects
    custom_objects = {
        'CastToFloat32': CastToFloat32,
        'masked_mse_loss': masked_mse_loss,
        'masked_mae_loss': masked_mae_loss
    }
    model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)
    print(f"   ✓ Model loaded: {model_path}")
    
    # Load test data
    print("\n2. Loading test data...")
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
    print(f"   ✓ Test sequences: {len(X_test)}")
    
    # Preprocess
    print("\n3. Preprocessing test data...")
    X_test_dual, y_test_proc, test_meta = create_dual_channel_input(
        X_test, y_test,
        gap_fill_method='hybrid',
        temporal_max_days=3,
        spatial_max_dist=3
    )
    print(f"   ✓ Shape: {X_test_dual.shape}")
    
    # Generate predictions
    print("\n4. Generating predictions...")
    y_pred = model.predict(X_test_dual, verbose=0)
    print(f"   ✓ Predictions shape: {y_pred.shape}")
    
    # Create output directory
    output_dir = Path('mc_lstm_forecasting/prediction_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Plot individual predictions (select 8 diverse examples)
    print("\n5. Creating individual prediction comparisons...")
    n_examples = min(8, len(X_test))
    
    # Select diverse examples: early, mid, late, high/low risk
    indices = [
        0,  # First
        len(X_test) // 4,  # Early
        len(X_test) // 2,  # Middle
        3 * len(X_test) // 4,  # Late
        len(X_test) - 1,  # Last
    ]
    
    # Add examples with highest and lowest mean predictions
    mean_preds = y_pred.mean(axis=(1, 2, 3))
    indices.append(np.argmax(mean_preds))  # Highest risk
    indices.append(np.argmin(mean_preds))  # Lowest risk
    
    # Add one with highest error
    errors = np.abs(y_test_proc - y_pred).mean(axis=(1, 2, 3))
    indices.append(np.argmax(errors))
    
    indices = sorted(set(indices))[:n_examples]
    
    for i, idx in enumerate(indices):
        save_path = output_dir / f'prediction_{idx:03d}.png'
        plot_prediction_comparison(
            X_test_dual[idx],
            y_test_proc[idx],
            y_pred[idx],
            idx,
            save_path
        )
    
    # Overall scatter analysis
    print("\n6. Creating scatter analysis...")
    plot_scatter_analysis(
        y_test_proc,
        y_pred,
        output_dir / 'scatter_analysis.png'
    )
    
    # Spatial error analysis
    print("\n7. Creating spatial error analysis...")
    plot_spatial_error_analysis(
        y_test_proc,
        y_pred,
        output_dir / 'spatial_error_analysis.png'
    )
    
    # Calculate overall statistics
    print("\n8. Computing overall statistics...")
    valid_mask = y_test_proc.flatten() != SENTINEL_VALUE
    y_true_valid = y_test_proc.flatten()[valid_mask]
    y_pred_valid = y_pred.flatten()[valid_mask]
    
    mse = np.mean((y_true_valid - y_pred_valid)**2)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    rmse = np.sqrt(mse)
    
    from scipy.stats import pearsonr
    r, p_value = pearsonr(y_true_valid, y_pred_valid)
    r_squared = r**2
    
    stats = {
        'n_sequences': len(X_test),
        'n_valid_pixels': int(valid_mask.sum()),
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r_squared': float(r_squared),
        'pearson_r': float(r),
        'p_value': float(p_value),
        'mean_prediction': float(y_pred_valid.mean()),
        'mean_ground_truth': float(y_true_valid.mean()),
        'std_prediction': float(y_pred_valid.std()),
        'std_ground_truth': float(y_true_valid.std())
    }
    
    # Save statistics
    with open(output_dir / 'prediction_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}/")
    print(f"  • {len(indices)} individual prediction comparisons")
    print(f"  • scatter_analysis.png - Prediction quality analysis")
    print(f"  • spatial_error_analysis.png - Spatial error distribution")
    print(f"  • prediction_statistics.json - Overall metrics")
    print(f"\nOverall Test Set Performance:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r_squared:.6f}")
    print(f"  Valid pixels: {valid_mask.sum():,}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
