"""
Data preprocessing utilities for MC probability forecasting.

Includes gap-filling strategies that avoid data leakage.
"""

import numpy as np
import logging
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


SENTINEL_VALUE = -1.0  # Value outside [0, 1] range to indicate missing data


def fill_with_sentinel(data: np.ndarray, sentinel: float = SENTINEL_VALUE) -> np.ndarray:
    """
    Replace NaN values with sentinel value.
    
    This allows the model to distinguish between:
    - Missing data (sentinel = -1)
    - Low bloom probability (valid value ≈ 0)
    
    Args:
        data: Array with NaN values
        sentinel: Value to use for missing data (default: -1.0)
        
    Returns:
        Array with NaN replaced by sentinel
    """
    filled = data.copy()
    filled[np.isnan(filled)] = sentinel
    return filled


def temporal_forward_fill(
    sequence: np.ndarray,
    max_fill_days: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill missing values using temporal persistence (last valid value).
    
    Only looks BACKWARD in time - NO DATA LEAKAGE.
    
    Strategy:
    - For each pixel, carry forward the last valid value
    - Maximum carry-forward distance: max_fill_days
    - Remaining gaps left as NaN
    
    Args:
        sequence: (T, H, W, C) temporal sequence with NaN for missing
        max_fill_days: Maximum days to carry forward a value
        
    Returns:
        filled_sequence: Same shape, NaN filled where possible
        fill_mask: Boolean array (True = pixel was filled)
    """
    T, H, W, C = sequence.shape
    filled = sequence.copy()
    fill_mask = np.zeros_like(sequence, dtype=bool)
    
    # Process each spatial location and channel separately
    for h in range(H):
        for w in range(W):
            for c in range(C):
                last_valid_value = None
                days_since_valid = 0
                
                for t in range(T):
                    value = sequence[t, h, w, c]
                    
                    if np.isfinite(value):
                        # Valid observation - use it
                        last_valid_value = value
                        days_since_valid = 0
                    elif last_valid_value is not None and days_since_valid < max_fill_days:
                        # Fill with persistence
                        filled[t, h, w, c] = last_valid_value
                        fill_mask[t, h, w, c] = True
                        days_since_valid += 1
                    else:
                        # Can't fill - leave as NaN
                        days_since_valid += 1
    
    return filled, fill_mask


def spatial_interpolation(
    map_2d: np.ndarray,
    method: str = 'nearest',
    max_distance: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill missing pixels using spatial interpolation from valid neighbors.
    
    Only uses same-timestep data - NO DATA LEAKAGE.
    
    Args:
        map_2d: (H, W) spatial map with NaN for missing
        method: Interpolation method ('nearest', 'linear')
        max_distance: Maximum pixel distance for interpolation
        
    Returns:
        filled_map: Same shape, NaN filled where possible
        confidence: Float array (1.0=original, <1.0=interpolated)
    """
    filled = map_2d.copy()
    valid_mask = np.isfinite(map_2d)
    
    if not np.any(valid_mask):
        # No valid data - can't interpolate
        return filled, np.zeros_like(filled)
    
    if not np.any(~valid_mask):
        # No missing data - nothing to fill
        return filled, np.ones_like(filled)
    
    # Get coordinates
    H, W = map_2d.shape
    y_valid, x_valid = np.where(valid_mask)
    y_missing, x_missing = np.where(~valid_mask)
    
    # Calculate distance to nearest valid pixel
    dist_map = distance_transform_edt(~valid_mask)
    
    # Prepare interpolation
    values_valid = map_2d[valid_mask]
    points_valid = np.column_stack([y_valid, x_valid])
    points_missing = np.column_stack([y_missing, x_missing])
    
    # Only interpolate pixels within max_distance
    close_enough = dist_map[~valid_mask] <= max_distance
    
    if np.any(close_enough):
        points_to_fill = points_missing[close_enough]
        
        try:
            # Interpolate
            interp_values = griddata(
                points_valid, 
                values_valid,
                points_to_fill,
                method=method,
                fill_value=np.nan
            )
            
            # Clip to valid probability range and update
            interp_values = np.clip(interp_values, 0, 1)
            
            # Update filled array
            for i, (y, x) in enumerate(points_to_fill):
                if np.isfinite(interp_values[i]):
                    filled[y, x] = interp_values[i]
        
        except Exception as e:
            logging.warning(f"Spatial interpolation failed: {e}")
    
    # Calculate confidence (1.0 for original, decreases with distance)
    confidence = np.ones_like(filled)
    confidence[~valid_mask] = np.maximum(0, 1 - dist_map[~valid_mask] / max_distance)
    confidence[~np.isfinite(filled)] = 0
    
    return filled, confidence


def hybrid_gap_fill(
    sequence: np.ndarray,
    temporal_max_days: int = 3,
    spatial_max_dist: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid gap-filling: temporal persistence + spatial interpolation.
    
    Strategy (hierarchical, most reliable first):
    1. Temporal forward-fill (use last valid value, up to max_days)
    2. Spatial interpolation (from nearby valid pixels)
    3. Remaining gaps left as NaN
    
    Args:
        sequence: (T, H, W, C) temporal sequence
        temporal_max_days: Max days for temporal persistence
        spatial_max_dist: Max pixel distance for spatial interpolation
        
    Returns:
        filled_sequence: Same shape, gaps filled where possible
        fill_type: Integer array indicating fill method:
                   0 = original data
                   1 = temporal fill
                   2 = spatial fill
                   -1 = unfilled (still NaN)
    """
    T, H, W, C = sequence.shape
    filled = sequence.copy()
    fill_type = np.zeros((T, H, W, C), dtype=int)
    
    # Step 1: Temporal forward-fill
    logging.debug("Applying temporal forward-fill...")
    filled_temporal, temporal_mask = temporal_forward_fill(
        filled, 
        max_fill_days=temporal_max_days
    )
    filled = filled_temporal
    fill_type[temporal_mask] = 1
    
    # Step 2: Spatial interpolation for remaining gaps
    logging.debug("Applying spatial interpolation...")
    for t in range(T):
        for c in range(C):
            map_2d = filled[t, :, :, c]
            still_missing = np.isnan(map_2d)
            
            if np.any(still_missing) and np.any(np.isfinite(map_2d)):
                # Apply spatial interpolation
                filled_spatial, _ = spatial_interpolation(
                    map_2d,
                    method='nearest',
                    max_distance=spatial_max_dist
                )
                
                # Mark newly filled pixels
                newly_filled = np.isfinite(filled_spatial) & still_missing
                filled[t, :, :, c][newly_filled] = filled_spatial[newly_filled]
                fill_type[t, :, :, c][newly_filled] = 2
    
    # Mark remaining unfilled
    fill_type[np.isnan(filled)] = -1
    
    return filled, fill_type


def preprocess_for_training(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'sentinel',
    **kwargs
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess data for training with specified gap-filling method.
    
    Args:
        X: Input sequences (N, T, H, W, C) with NaN for missing
        y: Target maps (N, H, W, C) with NaN for missing
        method: Gap-filling method:
                'sentinel' - Replace NaN with -1.0
                'temporal' - Temporal forward-fill + sentinel
                'hybrid' - Temporal + spatial + sentinel
        **kwargs: Additional arguments for gap-filling methods
        
    Returns:
        X_processed: Processed input sequences
        y_processed: Processed targets
        metadata: Dict with preprocessing info
    """
    logging.info(f"Preprocessing data with method: {method}")
    
    metadata = {
        'method': method,
        'original_nan_count_X': np.isnan(X).sum(),
        'original_nan_count_y': np.isnan(y).sum()
    }
    
    if method == 'sentinel':
        # Simple: just replace NaN with sentinel
        X_processed = fill_with_sentinel(X)
        y_processed = fill_with_sentinel(y)
        metadata['fill_type'] = None
        
    elif method == 'temporal':
        # Temporal forward-fill + sentinel for remaining
        X_filled = []
        fill_types = []
        
        for i in range(len(X)):
            filled_seq, fill_type = temporal_forward_fill(
                X[i],
                max_fill_days=kwargs.get('temporal_max_days', 3)
            )
            X_filled.append(filled_seq)
            fill_types.append(fill_type)
        
        X_filled = np.array(X_filled)
        fill_types = np.array(fill_types)
        
        # Replace remaining NaN with sentinel
        X_processed = fill_with_sentinel(X_filled)
        y_processed = fill_with_sentinel(y)
        
        metadata['fill_type'] = 'temporal'
        metadata['temporal_filled_count'] = fill_types.sum()
        
    elif method == 'hybrid':
        # Temporal + spatial + sentinel for remaining
        X_filled = []
        fill_types = []
        
        for i in range(len(X)):
            filled_seq, fill_type = hybrid_gap_fill(
                X[i],
                temporal_max_days=kwargs.get('temporal_max_days', 3),
                spatial_max_dist=kwargs.get('spatial_max_dist', 3)
            )
            X_filled.append(filled_seq)
            fill_types.append(fill_type)
        
        X_filled = np.array(X_filled)
        fill_types = np.array(fill_types)
        
        # Replace remaining NaN with sentinel
        X_processed = fill_with_sentinel(X_filled)
        y_processed = fill_with_sentinel(y)
        
        metadata['fill_type'] = fill_types
        metadata['temporal_filled_count'] = (fill_types == 1).sum()
        metadata['spatial_filled_count'] = (fill_types == 2).sum()
        metadata['unfilled_count'] = (fill_types == -1).sum()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    metadata['final_nan_count_X'] = np.isnan(X_processed).sum()
    metadata['final_nan_count_y'] = np.isnan(y_processed).sum()
    
    logging.info(f"Preprocessing complete:")
    logging.info(f"  Original NaN count: {metadata['original_nan_count_X']:,}")
    logging.info(f"  Final NaN count: {metadata['final_nan_count_X']:,}")
    
    if method in ['temporal', 'hybrid']:
        logging.info(f"  Temporal fills: {metadata.get('temporal_filled_count', 0):,}")
        if method == 'hybrid':
            logging.info(f"  Spatial fills: {metadata.get('spatial_filled_count', 0):,}")
            logging.info(f"  Remaining unfilled: {metadata.get('unfilled_count', 0):,}")
    
    return X_processed, y_processed, metadata


def create_dual_channel_input(X, y, gap_fill_method='hybrid', **kwargs):
    """
    Create dual-channel input for Phase 2 model: [probability, validity_mask].
    
    This function prepares data for the dual-channel ConvLSTM architecture, which
    explicitly separates MC probability values from their validity status. The model
    can learn different patterns for original data vs gap-filled data.
    
    Args:
        X: Input sequences, shape (N, timesteps, height, width)
        y: Target maps, shape (N, height, width)
        gap_fill_method: Gap-filling strategy
            - 'hybrid': Temporal forward-fill + spatial interpolation (recommended)
            - 'temporal': Only temporal forward-fill
            - 'sentinel': No gap-filling, use sentinel only
        **kwargs: Additional arguments for gap-filling methods
            - temporal_max_days: Max days for temporal fill (default: 3)
            - spatial_max_dist: Max pixels for spatial fill (default: 3)
    
    Returns:
        X_dual: Dual-channel input, shape (N, timesteps, height, width, 2)
                Channel 0: MC probability (gap-filled)
                Channel 1: Validity mask (1=original, 0=filled, -1=unfilled)
        y_processed: Processed target, shape (N, height, width)
        metadata: Dictionary with preprocessing statistics
    
    Channel Interpretation:
        - Channel 0 (Probability):
          * Original MC probabilities where available
          * Gap-filled values using specified method
          * Sentinel (-1.0) for remaining unfilled pixels
        
        - Channel 1 (Validity Mask):
          * 1.0 = Original satellite data (highest confidence)
          * 0.5 = Temporal forward-fill (medium confidence)
          * 0.25 = Spatial interpolation (lower confidence)
          * 0.0 = Sentinel/unfilled (no confidence)
    
    Example:
        >>> X_dual, y_proc, meta = create_dual_channel_input(X, y, method='hybrid')
        >>> print(X_dual.shape)  # (N, 5, 84, 73, 2)
        >>> print(f"Original pixels: {meta['original_pixel_count']:,}")
        >>> print(f"Temporal fills: {meta['temporal_filled_count']:,}")
        >>> print(f"Spatial fills: {meta['spatial_filled_count']:,}")
    """
    logging.info("=" * 80)
    logging.info("CREATING DUAL-CHANNEL INPUT FOR PHASE 2")
    logging.info(f"Gap-filling method: {gap_fill_method}")
    logging.info(f"Input shape: {X.shape}")
    logging.info("=" * 80)
    
    # Handle shape: remove last dimension if it's 1 (from data loader)
    if len(X.shape) == 5 and X.shape[-1] == 1:
        X = X.squeeze(-1)  # (N, timesteps, height, width, 1) -> (N, timesteps, height, width)
    if len(y.shape) == 4 and y.shape[-1] == 1:
        y = y.squeeze(-1)  # (N, height, width, 1) -> (N, height, width)
    
    N, timesteps, height, width = X.shape
    metadata = {
        'method': gap_fill_method,
        'original_nan_count_X': np.isnan(X).sum(),
        'original_nan_count_y': np.isnan(y).sum()
    }
    
    # Initialize channels
    probability_channel = np.zeros((N, timesteps, height, width), dtype=np.float32)
    validity_channel = np.zeros((N, timesteps, height, width), dtype=np.float32)
    
    if gap_fill_method == 'hybrid':
        # Apply hybrid gap-filling
        logging.info("Applying hybrid gap-filling (temporal + spatial)...")
        
        for i in range(N):
            # Add channel dimension for gap-filling function: (T,H,W) -> (T,H,W,1)
            seq_with_channel = X[i, :, :, :, np.newaxis]
            
            filled_seq, fill_type = hybrid_gap_fill(
                seq_with_channel,
                temporal_max_days=kwargs.get('temporal_max_days', 3),
                spatial_max_dist=kwargs.get('spatial_max_dist', 3)
            )
            
            # Remove channel dimension: (T,H,W,1) -> (T,H,W)
            filled_seq = filled_seq.squeeze(-1)
            fill_type = fill_type.squeeze(-1)
            
            # Channel 0: Filled probabilities
            probability_channel[i] = filled_seq
            
            # Channel 1: Validity mask based on fill type
            # 0 = original, 1 = temporal, 2 = spatial, -1 = unfilled
            validity_mask = np.zeros_like(fill_type, dtype=np.float32)
            validity_mask[fill_type == 0] = 1.0    # Original data
            validity_mask[fill_type == 1] = 0.5    # Temporal fill
            validity_mask[fill_type == 2] = 0.25   # Spatial fill
            validity_mask[fill_type == -1] = 0.0   # Unfilled
            validity_channel[i] = validity_mask
        
        # Replace remaining NaN with sentinel in probability channel
        probability_channel = fill_with_sentinel(probability_channel)
        
        # Count fill types
        metadata['original_pixel_count'] = (validity_channel == 1.0).sum()
        metadata['temporal_filled_count'] = (validity_channel == 0.5).sum()
        metadata['spatial_filled_count'] = (validity_channel == 0.25).sum()
        metadata['unfilled_count'] = (validity_channel == 0.0).sum()
        
    elif gap_fill_method == 'temporal':
        # Apply temporal forward-fill only
        logging.info("Applying temporal forward-fill...")
        
        for i in range(N):
            filled_seq, fill_mask = temporal_forward_fill(
                X[i],
                max_fill_days=kwargs.get('temporal_max_days', 3)
            )
            
            # Channel 0: Filled probabilities
            probability_channel[i] = filled_seq
            
            # Channel 1: Validity mask (1=original, 0.5=temporal fill)
            validity_mask = np.where(fill_mask, 1.0, 0.5)
            validity_mask[np.isnan(filled_seq)] = 0.0  # Unfilled pixels
            validity_channel[i] = validity_mask
        
        # Replace remaining NaN with sentinel
        probability_channel = fill_with_sentinel(probability_channel)
        
        metadata['original_pixel_count'] = (validity_channel == 1.0).sum()
        metadata['temporal_filled_count'] = (validity_channel == 0.5).sum()
        metadata['unfilled_count'] = (validity_channel == 0.0).sum()
        
    elif gap_fill_method == 'sentinel':
        # No gap-filling, just sentinel
        logging.info("Using sentinel values (no gap-filling)...")
        
        probability_channel = X.copy()
        
        # Validity mask: 1 for original data, 0 for missing
        for i in range(N):
            validity_channel[i] = np.where(np.isnan(X[i]), 0.0, 1.0)
        
        # Replace NaN with sentinel
        probability_channel = fill_with_sentinel(probability_channel)
        
        metadata['original_pixel_count'] = (validity_channel == 1.0).sum()
        metadata['unfilled_count'] = (validity_channel == 0.0).sum()
        
    else:
        raise ValueError(f"Unknown gap-fill method: {gap_fill_method}")
    
    # Stack channels: (N, timesteps, height, width, 2)
    X_dual = np.stack([probability_channel, validity_channel], axis=-1)
    
    # Process target with sentinel - keep channel dimension for model compatibility
    y_processed = fill_with_sentinel(y)
    # Add channel dimension if needed: (N, H, W) -> (N, H, W, 1)
    if len(y_processed.shape) == 3:
        y_processed = y_processed[..., np.newaxis]
    
    metadata['final_shape'] = X_dual.shape
    metadata['final_nan_count_X'] = np.isnan(X_dual).sum()
    metadata['final_nan_count_y'] = np.isnan(y_processed).sum()
    
    # Summary statistics
    total_pixels = N * timesteps * height * width
    metadata['coverage_stats'] = {
        'original_pct': (metadata['original_pixel_count'] / total_pixels) * 100,
        'temporal_pct': (metadata.get('temporal_filled_count', 0) / total_pixels) * 100,
        'spatial_pct': (metadata.get('spatial_filled_count', 0) / total_pixels) * 100,
        'unfilled_pct': (metadata['unfilled_count'] / total_pixels) * 100
    }
    
    logging.info("Dual-channel preprocessing complete:")
    logging.info(f"  Output shape: {X_dual.shape}")
    logging.info(f"  Original pixels: {metadata['original_pixel_count']:,} ({metadata['coverage_stats']['original_pct']:.1f}%)")
    
    if gap_fill_method == 'hybrid':
        logging.info(f"  Temporal fills: {metadata['temporal_filled_count']:,} ({metadata['coverage_stats']['temporal_pct']:.1f}%)")
        logging.info(f"  Spatial fills: {metadata['spatial_filled_count']:,} ({metadata['coverage_stats']['spatial_pct']:.1f}%)")
    elif gap_fill_method == 'temporal':
        logging.info(f"  Temporal fills: {metadata['temporal_filled_count']:,} ({metadata['coverage_stats']['temporal_pct']:.1f}%)")
    
    logging.info(f"  Unfilled (sentinel): {metadata['unfilled_count']:,} ({metadata['coverage_stats']['unfilled_pct']:.1f}%)")
    logging.info(f"  Final NaN count: {metadata['final_nan_count_X']}")
    
    return X_dual, y_processed, metadata

