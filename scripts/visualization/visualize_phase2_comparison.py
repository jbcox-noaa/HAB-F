"""
Visualize the comparison between original Phase 2 and improved Phase 2.

This script creates comprehensive visualizations showing:
1. Training/validation loss curves
2. Learning rate schedules
3. Performance comparison bar charts
4. Improvement tracking over epochs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import re


def parse_log_file(log_path):
    """Parse training log to extract loss history."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Extract epoch-wise metrics
    lines = content.split('\n')
    for line in lines:
        # Match lines like: "7/7 ━━━ 9s 1s/step - lambda: 0.1354 - loss: 0.0535 - val_lambda: 0.1425 - val_loss: 0.0608 - learning_rate: 7.8323e-04"
        if 'val_loss:' in line and 'learning_rate:' in line:
            # Extract epoch number from previous "Epoch X/Y" line
            epoch_match = re.search(r'Epoch (\d+)/\d+', content[:content.find(line)])
            if epoch_match:
                epoch_matches = re.findall(r'Epoch (\d+)/\d+', content[:content.find(line)])
                if epoch_matches:
                    epoch = int(epoch_matches[-1])
                    
                    # Extract metrics
                    loss_match = re.search(r'loss: ([\d.]+)', line)
                    val_loss_match = re.search(r'val_loss: ([\d.]+)', line)
                    lr_match = re.search(r'learning_rate: ([\d.e-]+)', line)
                    
                    if loss_match and val_loss_match and lr_match:
                        epochs.append(epoch)
                        train_losses.append(float(loss_match.group(1)))
                        val_losses.append(float(val_loss_match.group(1)))
                        learning_rates.append(float(lr_match.group(1)))
    
    # Remove duplicates (keep last occurrence of each epoch)
    unique_data = {}
    for i, epoch in enumerate(epochs):
        unique_data[epoch] = {
            'train_loss': train_losses[i],
            'val_loss': val_losses[i],
            'lr': learning_rates[i]
        }
    
    epochs = sorted(unique_data.keys())
    train_losses = [unique_data[e]['train_loss'] for e in epochs]
    val_losses = [unique_data[e]['val_loss'] for e in epochs]
    learning_rates = [unique_data[e]['lr'] for e in epochs]
    
    return epochs, train_losses, val_losses, learning_rates


def load_csv_history(csv_path):
    """Load training history from CSV file."""
    df = pd.read_csv(csv_path)
    epochs = df['epoch'].values + 1  # Convert 0-indexed to 1-indexed
    train_losses = df['loss'].values
    val_losses = df['val_loss'].values
    learning_rates = df['learning_rate'].values if 'learning_rate' in df.columns else None
    return epochs, train_losses, val_losses, learning_rates


def main():
    """Create comprehensive comparison visualizations."""
    
    print("="*80)
    print("CREATING PHASE 2 COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Load data for original Phase 2
    print("\n1. Loading original Phase 2 data...")
    log_path_original = 'mc_lstm_forecasting/training_dual_channel_nohup.log'
    
    if Path(log_path_original).exists():
        epochs_orig, train_orig, val_orig, lr_orig = parse_log_file(log_path_original)
        print(f"   Original Phase 2: {len(epochs_orig)} epochs")
        print(f"   Best val_loss: {min(val_orig):.6f} at epoch {epochs_orig[np.argmin(val_orig)]}")
    else:
        print(f"   ⚠️  Original log not found: {log_path_original}")
        # Use known values
        epochs_orig = list(range(1, 13))
        val_orig = [0.0569, 0.0566, 0.0567, 0.0581, 0.0581, 0.0582, 
                    0.0607, 0.0604, 0.0611, 0.0625, 0.0640, 0.0634]
        train_orig = [0.0853, 0.0578, 0.0524, 0.0550, 0.0429, 0.0483,
                      0.0431, 0.0461, 0.0411, 0.0410, 0.0358, 0.0422]
        lr_orig = [5e-4] * 7 + [2.5e-4] * 5
    
    # Load data for improved Phase 2
    print("\n2. Loading improved Phase 2 data...")
    
    # Try CSV first
    csv_files = list(Path('mc_lstm_forecasting').glob('training_history_dual_v2_*.csv'))
    if csv_files:
        csv_path = sorted(csv_files)[-1]  # Get most recent
        epochs_improved, train_improved, val_improved, lr_improved = load_csv_history(csv_path)
        print(f"   Loaded from CSV: {csv_path}")
    else:
        # Parse log file
        log_path_improved = 'mc_lstm_forecasting/training_dual_channel_v3_improved.log'
        epochs_improved, train_improved, val_improved, lr_improved = parse_log_file(log_path_improved)
        print(f"   Loaded from log: {log_path_improved}")
    
    print(f"   Improved Phase 2: {len(epochs_improved)} epochs")
    print(f"   Best val_loss: {min(val_improved):.6f} at epoch {epochs_improved[np.argmin(val_improved)]}")
    
    # Load final results
    print("\n3. Loading final test results...")
    with open('mc_lstm_forecasting/results_dual_channel_v2.json', 'r') as f:
        results_improved = json.load(f)
    
    # Known results for other models
    results_baseline = {'test_mse': 0.094915, 'test_mae': 0.254631}
    results_phase1 = {'test_mse': 0.063398, 'test_mae': 0.219647}
    results_phase2_orig = {'test_mse': 0.063362, 'test_mae': 0.219019}
    
    # Create figure with subplots
    print("\n4. Creating visualizations...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Panel 1: Training Loss Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(epochs_orig, train_orig, 'o-', color='#e74c3c', 
             linewidth=2, markersize=6, label='Original Phase 2 (train)', alpha=0.7)
    ax1.plot(epochs_improved, train_improved, 's-', color='#3498db', 
             linewidth=2, markersize=4, label='Improved Phase 2 (train)', alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss: Original vs Improved Phase 2', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(max(epochs_orig), max(epochs_improved)) + 2)
    
    # Highlight overfitting in original
    ax1.axvline(x=2, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(2.5, max(train_orig) * 0.9, 'Original stopped\nimproving at epoch 2',
             fontsize=10, color='red', fontweight='bold')
    
    # ========================================================================
    # Panel 2: Validation Loss Comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, :2])
    
    ax2.plot(epochs_orig, val_orig, 'o-', color='#e74c3c', 
             linewidth=3, markersize=8, label='Original Phase 2', alpha=0.8)
    ax2.plot(epochs_improved, val_improved, 's-', color='#2ecc71', 
             linewidth=3, markersize=5, label='Improved Phase 2', alpha=0.8)
    
    # Mark best epochs
    best_epoch_orig = epochs_orig[np.argmin(val_orig)]
    best_val_orig = min(val_orig)
    best_epoch_improved = epochs_improved[np.argmin(val_improved)]
    best_val_improved = min(val_improved)
    
    ax2.plot(best_epoch_orig, best_val_orig, 'r*', markersize=20, 
             label=f'Best Original: {best_val_orig:.4f} @ epoch {best_epoch_orig}')
    ax2.plot(best_epoch_improved, best_val_improved, 'g*', markersize=20,
             label=f'Best Improved: {best_val_improved:.4f} @ epoch {best_epoch_improved}')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss: Original vs Improved Phase 2\n(Lower is Better)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(max(epochs_orig), max(epochs_improved)) + 2)
    
    # Add improvement annotation
    improvement = ((best_val_orig - best_val_improved) / best_val_orig) * 100
    ax2.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
             transform=ax2.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             verticalalignment='top')
    
    # ========================================================================
    # Panel 3: Learning Rate Schedules
    # ========================================================================
    ax3 = fig.add_subplot(gs[2, :2])
    
    if lr_orig is not None and lr_improved is not None:
        ax3.plot(epochs_orig, lr_orig, 'o-', color='#e74c3c', 
                linewidth=2, markersize=6, label='Original (ReduceLROnPlateau)', alpha=0.7)
        ax3.plot(epochs_improved, lr_improved, 's-', color='#9b59b6', 
                linewidth=2, markersize=4, label='Improved (Cosine Annealing)', alpha=0.7)
        
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax3.set_title('Learning Rate Schedule Comparison', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.set_yscale('log')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate key differences
        ax3.axvline(x=7, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax3.text(7.5, max(lr_orig) * 0.5, 'Original LR\ndropped at epoch 7',
                fontsize=9, color='red')
    
    # ========================================================================
    # Panel 4: Performance Bar Chart (MSE)
    # ========================================================================
    ax4 = fig.add_subplot(gs[0, 2])
    
    models = ['Baseline', 'Phase 1', 'Phase 2\nOriginal', 'Phase 2\nImproved']
    mse_values = [
        results_baseline['test_mse'],
        results_phase1['test_mse'],
        results_phase2_orig['test_mse'],
        results_improved['test_mse']
    ]
    
    colors = ['#95a5a6', '#f39c12', '#e74c3c', '#2ecc71']
    bars = ax4.bar(models, mse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mse_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Test MSE', fontsize=11, fontweight='bold')
    ax4.set_title('Test MSE Comparison\n(Lower is Better)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, max(mse_values) * 1.2)
    
    # ========================================================================
    # Panel 5: Performance Bar Chart (MAE)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    mae_values = [
        results_baseline['test_mae'],
        results_phase1['test_mae'],
        results_phase2_orig['test_mae'],
        results_improved['test_mae']
    ]
    
    bars = ax5.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax5.set_ylabel('Test MAE', fontsize=11, fontweight='bold')
    ax5.set_title('Test MAE Comparison\n(Lower is Better)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax5.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax5.set_ylim(0, max(mae_values) * 1.2)
    
    # ========================================================================
    # Panel 6: Improvement Percentages
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    
    improvements = [
        0,  # Baseline
        ((results_baseline['test_mse'] - results_phase1['test_mse']) / results_baseline['test_mse']) * 100,
        ((results_baseline['test_mse'] - results_phase2_orig['test_mse']) / results_baseline['test_mse']) * 100,
        ((results_baseline['test_mse'] - results_improved['test_mse']) / results_baseline['test_mse']) * 100
    ]
    
    bars = ax6.bar(models, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax6.set_ylabel('Improvement vs Baseline (%)', fontsize=11, fontweight='bold')
    ax6.set_title('MSE Improvement Over Baseline\n(Higher is Better)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax6.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax6.set_ylim(0, max(improvements) * 1.15)
    ax6.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    
    # Add main title
    fig.suptitle('Phase 2 Comparison: Original vs Improved\n' + 
                 'Impact of Regularization, Augmentation, and Better LR Schedule',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = 'mc_lstm_forecasting/phase2_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Create second figure: Detailed improvement tracking
    print("\n5. Creating improvement tracking visualization...")
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 7: Cumulative improvement over epochs
    baseline_val = val_orig[0]  # Use first epoch as baseline
    improvement_orig = [((baseline_val - v) / baseline_val) * 100 for v in val_orig]
    improvement_improved = [((baseline_val - v) / baseline_val) * 100 for v in val_improved]
    
    ax7.plot(epochs_orig, improvement_orig, 'o-', color='#e74c3c', 
             linewidth=3, markersize=8, label='Original Phase 2', alpha=0.8)
    ax7.plot(epochs_improved, improvement_improved, 's-', color='#2ecc71', 
             linewidth=3, markersize=5, label='Improved Phase 2', alpha=0.8)
    
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax7.fill_between(epochs_improved, 0, improvement_improved, 
                     color='green', alpha=0.1)
    
    ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Improvement vs Initial (%)', fontsize=12, fontweight='bold')
    ax7.set_title('Cumulative Improvement Over Training', 
                  fontsize=14, fontweight='bold', pad=15)
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 8: Key differences table
    ax8.axis('off')
    
    table_data = [
        ['Metric', 'Original', 'Improved', 'Change'],
        ['Best Epoch', f'{best_epoch_orig}', f'{best_epoch_improved}', 
         f'+{best_epoch_improved - best_epoch_orig}'],
        ['Best Val Loss', f'{best_val_orig:.5f}', f'{best_val_improved:.5f}', 
         f'{improvement:.1f}%↓'],
        ['Test MSE', f'{results_phase2_orig["test_mse"]:.5f}', 
         f'{results_improved["test_mse"]:.5f}', 
         f'{results_improved["improvement_vs_phase2_original"]:.1f}%↓'],
        ['Test MAE', f'{results_phase2_orig["test_mae"]:.5f}', 
         f'{results_improved["test_mae"]:.5f}', 
         f'{((results_phase2_orig["test_mae"] - results_improved["test_mae"]) / results_phase2_orig["test_mae"] * 100):.1f}%↓'],
        ['Dropout', '0.2', '0.4', '+0.2'],
        ['L2 Reg', 'None', '0.001', 'Added'],
        ['Augmentation', 'No', 'Yes', 'Enabled'],
        ['LR Schedule', 'ReduceLR', 'Cosine', 'Changed'],
        ['Initial LR', '5e-4', '1e-3', '2x'],
    ]
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_edgecolor('#bdc3c7')
    
    # Highlight improvement rows
    for i in [1, 2, 3, 4]:
        table[(i, 3)].set_facecolor('#d5f4e6')
        table[(i, 3)].set_text_props(weight='bold', color='#27ae60')
    
    ax8.set_title('Key Improvements Summary', fontsize=14, fontweight='bold', pad=20)
    
    fig2.suptitle('Phase 2 Improvement Tracking and Key Metrics',
                  fontsize=16, fontweight='bold')
    
    output_path2 = 'mc_lstm_forecasting/phase2_improvement_tracking.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Improvement tracking saved: {output_path2}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {output_path}")
    print(f"  2. {output_path2}")
    print(f"\nKey Findings:")
    print(f"  • Best epoch: {best_epoch_orig} → {best_epoch_improved} ({best_epoch_improved - best_epoch_orig}x more training)")
    print(f"  • Val loss: {best_val_orig:.5f} → {best_val_improved:.5f} ({improvement:.1f}% better)")
    print(f"  • Test MSE: {results_phase2_orig['test_mse']:.5f} → {results_improved['test_mse']:.5f} ({results_improved['improvement_vs_phase2_original']:.1f}% better)")
    print(f"  • Overall improvement vs baseline: {improvements[2]:.1f}% → {improvements[3]:.1f}%")
    
    plt.show()


if __name__ == '__main__':
    main()
