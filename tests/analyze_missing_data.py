"""
Deep analysis of missing data patterns and impact on MC forecasting model.

This script investigates:
1. How much data is actually missing (lake pixels with NaN)
2. Spatial and temporal patterns of missing data
3. Impact of zero-filling on model predictions
4. Whether 0 is a valid MC probability that we're corrupting
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from mc_lstm_forecasting.utils import load_mc_sequences, configure_logging


def analyze_missing_data_patterns():
    """Comprehensive analysis of missing data in MC probability maps."""
    
    configure_logging()
    
    print("="*80)
    print("MISSING DATA ANALYSIS FOR MC FORECASTING")
    print("="*80)
    
    # Load data WITHOUT preprocessing (keep NaN values)
    print("\nLoading raw data (with NaN values)...")
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()
    
    # Get all splits combined
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    all_dates = metadata['train_dates'] + metadata['val_dates'] + metadata['test_dates']
    
    print(f"\nTotal sequences: {len(X_all)}")
    print(f"Sequence shape: {X_all.shape}")
    print(f"Target shape: {y_all.shape}")
    
    # Define lake mask (pixels that are EVER valid)
    print("\n" + "="*80)
    print("1. IDENTIFYING LAKE VS NON-LAKE PIXELS")
    print("="*80)
    
    # Lake pixels are those that have at least one valid value across all data
    lake_mask = np.zeros((84, 73), dtype=bool)
    for seq in X_all:
        for t in range(seq.shape[0]):
            valid = np.isfinite(seq[t, :, :, 0])
            lake_mask |= valid
    
    for target in y_all:
        valid = np.isfinite(target[:, :, 0])
        lake_mask |= valid
    
    n_lake_pixels = lake_mask.sum()
    n_total_pixels = 84 * 73
    n_non_lake = n_total_pixels - n_lake_pixels
    
    print(f"\nTotal pixels: {n_total_pixels:,}")
    print(f"Lake pixels: {n_lake_pixels:,} ({100*n_lake_pixels/n_total_pixels:.1f}%)")
    print(f"Non-lake pixels: {n_non_lake:,} ({100*n_non_lake/n_total_pixels:.1f}%)")
    
    # Analyze missing data WITHIN lake pixels
    print("\n" + "="*80)
    print("2. MISSING DATA WITHIN LAKE PIXELS")
    print("="*80)
    
    # For each sequence, count missing lake pixels in each timestep
    missing_stats = {
        'input_t1': [], 'input_t2': [], 'input_t3': [], 'input_t4': [], 'input_t5': [],
        'target': []
    }
    
    for i, (X_seq, y_target) in enumerate(zip(X_all, y_all)):
        for t in range(5):
            data = X_seq[t, :, :, 0]
            lake_data = data[lake_mask]
            n_missing = np.isnan(lake_data).sum()
            pct_missing = 100 * n_missing / n_lake_pixels
            missing_stats[f'input_t{t+1}'].append(pct_missing)
        
        target_data = y_target[:, :, 0][lake_mask]
        n_missing = np.isnan(target_data).sum()
        pct_missing = 100 * n_missing / n_lake_pixels
        missing_stats['target'].append(pct_missing)
    
    # Print statistics
    print("\nMissing data percentage in lake pixels:")
    print(f"{'Timestep':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    for key in ['input_t1', 'input_t2', 'input_t3', 'input_t4', 'input_t5', 'target']:
        values = np.array(missing_stats[key])
        print(f"{key:<15} {values.mean():>8.1f}% {np.median(values):>8.1f}% "
              f"{values.min():>8.1f}% {values.max():>8.1f}%")
    
    # Count sequences with ANY missing lake pixels
    sequences_with_gaps = 0
    for i, (X_seq, y_target) in enumerate(zip(X_all, y_all)):
        has_gap = False
        for t in range(5):
            data = X_seq[t, :, :, 0]
            lake_data = data[lake_mask]
            if np.any(np.isnan(lake_data)):
                has_gap = True
                break
        
        target_data = y_target[:, :, 0][lake_mask]
        if np.any(np.isnan(target_data)):
            has_gap = True
        
        if has_gap:
            sequences_with_gaps += 1
    
    print(f"\nSequences with at least one missing lake pixel: {sequences_with_gaps}/{len(X_all)} "
          f"({100*sequences_with_gaps/len(X_all):.1f}%)")
    
    # Analyze distribution of actual MC probabilities
    print("\n" + "="*80)
    print("3. DISTRIBUTION OF VALID MC PROBABILITIES")
    print("="*80)
    
    all_valid_probs = []
    for y_target in y_all:
        target_data = y_target[:, :, 0][lake_mask]
        valid_probs = target_data[np.isfinite(target_data)]
        all_valid_probs.extend(valid_probs)
    
    all_valid_probs = np.array(all_valid_probs)
    
    print(f"\nTotal valid lake pixel values: {len(all_valid_probs):,}")
    print(f"Min probability: {all_valid_probs.min():.6f}")
    print(f"Max probability: {all_valid_probs.max():.6f}")
    print(f"Mean probability: {all_valid_probs.mean():.6f}")
    print(f"Median probability: {np.median(all_valid_probs):.6f}")
    
    # CRITICAL: How many valid probabilities are near zero?
    near_zero = (all_valid_probs < 0.1).sum()
    very_near_zero = (all_valid_probs < 0.05).sum()
    exactly_zero = (all_valid_probs == 0.0).sum()
    
    print(f"\nProbabilities < 0.1: {near_zero:,} ({100*near_zero/len(all_valid_probs):.1f}%)")
    print(f"Probabilities < 0.05: {very_near_zero:,} ({100*very_near_zero/len(all_valid_probs):.1f}%)")
    print(f"Probabilities == 0.0: {exactly_zero:,} ({100*exactly_zero/len(all_valid_probs):.1f}%)")
    
    print("\n⚠️  CRITICAL FINDING:")
    if near_zero > 0.1 * len(all_valid_probs):
        print(f"   {100*near_zero/len(all_valid_probs):.1f}% of valid probabilities are < 0.1")
        print("   Using 0 as a fill value CONFLICTS with real low-probability areas!")
        print("   This is causing the model to confuse 'missing data' with 'no bloom'")
    
    # Spatial patterns of missing data
    print("\n" + "="*80)
    print("4. SPATIAL PATTERNS OF MISSING DATA")
    print("="*80)
    
    # Calculate how often each lake pixel is missing
    missing_frequency = np.zeros((84, 73))
    n_observations = np.zeros((84, 73))
    
    for X_seq in X_all:
        for t in range(5):
            data = X_seq[t, :, :, 0]
            for i in range(84):
                for j in range(73):
                    if lake_mask[i, j]:
                        n_observations[i, j] += 1
                        if np.isnan(data[i, j]):
                            missing_frequency[i, j] += 1
    
    for y_target in y_all:
        data = y_target[:, :, 0]
        for i in range(84):
            for j in range(73):
                if lake_mask[i, j]:
                    n_observations[i, j] += 1
                    if np.isnan(data[i, j]):
                        missing_frequency[i, j] += 1
    
    # Convert to percentage
    missing_pct = np.zeros((84, 73))
    missing_pct[lake_mask] = 100 * missing_frequency[lake_mask] / n_observations[lake_mask]
    
    print(f"\nMissing data frequency across lake:")
    print(f"  Min: {missing_pct[lake_mask].min():.1f}%")
    print(f"  Max: {missing_pct[lake_mask].max():.1f}%")
    print(f"  Mean: {missing_pct[lake_mask].mean():.1f}%")
    print(f"  Median: {np.median(missing_pct[lake_mask]):.1f}%")
    
    # Temporal patterns
    print("\n" + "="*80)
    print("5. TEMPORAL PATTERNS OF MISSING DATA")
    print("="*80)
    
    # Group by month
    monthly_missing = {}
    for i, date_str in enumerate(all_dates):
        month = date_str[:6]  # YYYYMM
        if month not in monthly_missing:
            monthly_missing[month] = []
        
        # Check target
        target_data = y_all[i, :, :, 0][lake_mask]
        pct_missing = 100 * np.isnan(target_data).sum() / n_lake_pixels
        monthly_missing[month].append(pct_missing)
    
    print(f"\nMissing data by month:")
    print(f"{'Month':<10} {'Mean Missing %':<15} {'N Sequences':<15}")
    print("-" * 40)
    for month in sorted(monthly_missing.keys()):
        values = monthly_missing[month]
        print(f"{month:<10} {np.mean(values):>13.1f}% {len(values):>13}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("6. CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = 'mc_lstm_forecasting/missing_data_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Distribution of MC probabilities
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(all_valid_probs, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (fill value)')
    axes[0].set_xlabel('MC Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Valid MC Probabilities', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_valid_probs, bins=50, color='blue', alpha=0.7, edgecolor='black', 
                 cumulative=True, density=True)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (fill value)')
    axes[1].axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('MC Probability', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mc_probability_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/mc_probability_distribution.png")
    plt.close()
    
    # Plot 2: Spatial pattern of missing data
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Mask non-lake pixels
    plot_data = missing_pct.copy()
    plot_data[~lake_mask] = np.nan
    
    im = ax.imshow(plot_data, cmap='RdYlGn_r', vmin=0, vmax=100, interpolation='nearest')
    ax.set_title('Percentage of Missing Data by Location', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% Missing', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spatial_missing_pattern.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/spatial_missing_pattern.png")
    plt.close()
    
    # Plot 3: Missing data over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # By timestep
    timesteps = ['Input t-4', 'Input t-3', 'Input t-2', 'Input t-1', 'Input t0', 'Target']
    means = [np.mean(missing_stats[k]) for k in missing_stats.keys()]
    
    axes[0].bar(timesteps, means, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Mean % Missing (Lake Pixels)', fontsize=12)
    axes[0].set_title('Missing Data by Timestep in Sequence', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (x, y) in enumerate(zip(timesteps, means)):
        axes[0].text(i, y + 1, f'{y:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # By month
    months = sorted(monthly_missing.keys())
    month_means = [np.mean(monthly_missing[m]) for m in months]
    
    axes[1].plot(range(len(months)), month_means, 'o-', linewidth=2, markersize=8, color='darkred')
    axes[1].set_xticks(range(len(months)))
    axes[1].set_xticklabels(months, rotation=45, ha='right')
    axes[1].set_ylabel('Mean % Missing (Lake Pixels)', fontsize=12)
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_title('Missing Data by Month', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_missing_pattern.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/temporal_missing_pattern.png")
    plt.close()
    
    # Final summary
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    print("\n1. MISSING DATA PREVALENCE:")
    print(f"   - {100*sequences_with_gaps/len(X_all):.1f}% of sequences have at least one missing lake pixel")
    print(f"   - Average {np.mean([np.mean(missing_stats[k]) for k in missing_stats.keys()]):.1f}% "
          "of lake pixels missing per timestep")
    
    print("\n2. ZERO-FILLING PROBLEM:")
    print(f"   - {100*near_zero/len(all_valid_probs):.1f}% of VALID probabilities are < 0.1")
    print(f"   - Using 0 as fill value creates ambiguity:")
    print("     * Model can't distinguish 'missing data' from 'low bloom probability'")
    print("     * This likely causes underprediction in low-bloom areas")
    
    print("\n3. SPATIAL PATTERNS:")
    print(f"   - Missing data varies from {missing_pct[lake_mask].min():.1f}% to "
          f"{missing_pct[lake_mask].max():.1f}% across lake")
    print("   - Some areas have more cloud cover / data gaps than others")
    
    print("\n4. TEMPORAL PATTERNS:")
    print("   - Missing data varies by season (cloud cover, ice, etc.)")
    print("   - Need gap-filling that respects temporal context")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)
    
    print("\n1. REPLACE ZERO-FILLING:")
    print("   Option A: Use sentinel value (e.g., -1) that's outside [0,1] range")
    print("   Option B: Separate validity mask channel")
    print("   Option C: Learned embedding for missing values")
    
    print("\n2. GAP-FILLING STRATEGIES (NO LEAKAGE):")
    print("   Option A: Temporal persistence (use last valid value)")
    print("   Option B: Spatial interpolation from nearby pixels")
    print("   Option C: Temporal interpolation from prior days only")
    print("   Option D: Learned imputation model trained separately")
    
    print("\n3. ARCHITECTURE IMPROVEMENTS:")
    print("   Option A: Attention mechanism to focus on valid pixels")
    print("   Option B: Separate pathway for mask information")
    print("   Option C: Loss function that weights valid pixels more")
    
    print("\n" + "="*80)
    
    return {
        'lake_mask': lake_mask,
        'n_lake_pixels': n_lake_pixels,
        'missing_stats': missing_stats,
        'all_valid_probs': all_valid_probs,
        'sequences_with_gaps': sequences_with_gaps,
        'missing_pct_spatial': missing_pct
    }


if __name__ == '__main__':
    results = analyze_missing_data_patterns()
