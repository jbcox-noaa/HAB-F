"""
Data augmentation functions for spectral patch sequences.
Priority 1 (Conservative): 24x augmentation via spatial flips, Gaussian noise, temporal shifts.
"""

import numpy as np
from typing import Tuple


def horizontal_flip(X: np.ndarray, mask_idx: int = -1) -> np.ndarray:
    """
    Flip patch horizontally (left-right).
    
    Args:
        X: Patch sequence (T, H, W, C) or (H, W, C)
        mask_idx: Index of mask channel (default: -1, last channel)
    
    Returns:
        Horizontally flipped patch
    """
    return np.flip(X, axis=-2)  # Flip along width dimension


def vertical_flip(X: np.ndarray, mask_idx: int = -1) -> np.ndarray:
    """
    Flip patch vertically (top-bottom).
    
    Args:
        X: Patch sequence (T, H, W, C) or (H, W, C)
        mask_idx: Index of mask channel (default: -1, last channel)
    
    Returns:
        Vertically flipped patch
    """
    return np.flip(X, axis=-3)  # Flip along height dimension


def add_gaussian_noise(
    X: np.ndarray, 
    sigma: float = 0.02, 
    mask_idx: int = -1,
    seed: int = None
) -> np.ndarray:
    """
    Add Gaussian noise to spectral bands only (not mask channel).
    
    Args:
        X: Patch sequence (T, H, W, C) where C includes spectral + mask
        sigma: Standard deviation of Gaussian noise (default: 0.02)
        mask_idx: Index of mask channel (default: -1, last channel)
        seed: Random seed for reproducibility
    
    Returns:
        Patch with added noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_noisy = X.copy()
    
    # Add noise to all channels except mask
    if mask_idx == -1:
        # Noise to all spectral bands
        noise = np.random.normal(0, sigma, X[..., :-1].shape)
        X_noisy[..., :-1] += noise
    else:
        # Add noise everywhere except mask_idx
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy += noise
        # Restore original mask (no noise on mask)
        X_noisy[..., mask_idx] = X[..., mask_idx]
    
    return X_noisy


def temporal_shift(
    X: np.ndarray,
    shift: int = 1,
    fill_mode: str = 'zero'
) -> np.ndarray:
    """
    Shift sequence in time by rolling or padding.
    
    Args:
        X: Patch sequence (T, H, W, C)
        shift: Number of timesteps to shift (+1 = later, -1 = earlier)
        fill_mode: How to fill shifted frames ('zero', 'edge', 'wrap')
    
    Returns:
        Temporally shifted sequence
    """
    T = X.shape[0]
    
    if fill_mode == 'wrap':
        # Circular shift (wrap around)
        return np.roll(X, shift, axis=0)
    
    elif fill_mode == 'edge':
        # Repeat edge frames
        X_shifted = np.roll(X, shift, axis=0)
        if shift > 0:
            X_shifted[:shift] = X_shifted[shift:shift+1]  # Repeat first valid frame
        elif shift < 0:
            X_shifted[shift:] = X_shifted[shift-1:shift]  # Repeat last valid frame
        return X_shifted
    
    else:  # 'zero'
        # Zero-pad (mask channel will indicate missing data)
        X_shifted = np.zeros_like(X)
        if shift > 0:
            X_shifted[shift:] = X[:-shift]
        elif shift < 0:
            X_shifted[:shift] = X[-shift:]
        else:
            X_shifted = X.copy()
        return X_shifted


def augment_patch_sequence(
    X: np.ndarray,
    y: float,
    augmentation_config: dict
) -> Tuple[np.ndarray, float]:
    """
    Apply random augmentations to a patch sequence.
    
    Args:
        X: Patch sequence (T, H, W, C)
        y: Binary label (unchanged by augmentation)
        augmentation_config: Dict with augmentation parameters
            {
                'h_flip_prob': 0.5,
                'v_flip_prob': 0.5,
                'noise_prob': 0.5,
                'noise_sigma': 0.02,
                'time_shift_prob': 0.3,
                'time_shift_range': (-1, 1)
            }
    
    Returns:
        Augmented (X, y) tuple
    """
    X_aug = X.copy()
    
    # Horizontal flip
    if np.random.rand() < augmentation_config.get('h_flip_prob', 0.5):
        X_aug = horizontal_flip(X_aug)
    
    # Vertical flip
    if np.random.rand() < augmentation_config.get('v_flip_prob', 0.5):
        X_aug = vertical_flip(X_aug)
    
    # Gaussian noise
    if np.random.rand() < augmentation_config.get('noise_prob', 0.5):
        sigma = augmentation_config.get('noise_sigma', 0.02)
        X_aug = add_gaussian_noise(X_aug, sigma=sigma)
    
    # Temporal shift
    if np.random.rand() < augmentation_config.get('time_shift_prob', 0.3):
        shift_range = augmentation_config.get('time_shift_range', (-1, 1))
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        if shift != 0:
            X_aug = temporal_shift(X_aug, shift=shift, fill_mode='zero')
    
    return X_aug, y


def get_priority1_config() -> dict:
    """
    Priority 1 (Conservative) augmentation configuration.
    
    Returns:
        Augmentation config dict with conservative parameters
    """
    return {
        'h_flip_prob': 0.5,       # 50% chance horizontal flip
        'v_flip_prob': 0.5,       # 50% chance vertical flip
        'noise_prob': 0.5,        # 50% chance Gaussian noise
        'noise_sigma': 0.02,      # Small noise (2% std)
        'time_shift_prob': 0.3,   # 30% chance temporal shift
        'time_shift_range': (-1, 1)  # ±1 day shift
    }


def get_priority2_config() -> dict:
    """
    Priority 2 (Moderate) augmentation configuration.
    
    Returns:
        Augmentation config dict with moderate parameters
    """
    return {
        'h_flip_prob': 0.7,
        'v_flip_prob': 0.7,
        'noise_prob': 0.6,
        'noise_sigma': 0.03,      # Slightly more noise
        'time_shift_prob': 0.5,
        'time_shift_range': (-2, 2)  # ±2 day shift
    }


if __name__ == '__main__':
    """Test augmentation functions."""
    # Create dummy patch sequence (14, 11, 11, 173)
    T, H, W, C = 14, 11, 11, 173
    X_test = np.random.randn(T, H, W, C).astype(np.float32)
    X_test[..., -1] = 1.0  # Set mask to all valid
    y_test = 1.0
    
    print("Testing augmentation functions...")
    print(f"Original shape: {X_test.shape}")
    
    # Test horizontal flip
    X_hflip = horizontal_flip(X_test)
    print(f"✓ Horizontal flip: {X_hflip.shape}")
    assert np.array_equal(X_hflip[:, :, 0, :], X_test[:, :, -1, :]), "H-flip failed"
    
    # Test vertical flip
    X_vflip = vertical_flip(X_test)
    print(f"✓ Vertical flip: {X_vflip.shape}")
    assert np.array_equal(X_vflip[:, 0, :, :], X_test[:, -1, :, :]), "V-flip failed"
    
    # Test Gaussian noise
    X_noise = add_gaussian_noise(X_test, sigma=0.02)
    print(f"✓ Gaussian noise: {X_noise.shape}")
    assert not np.array_equal(X_noise[..., :-1], X_test[..., :-1]), "Noise not added"
    assert np.array_equal(X_noise[..., -1], X_test[..., -1]), "Mask changed by noise"
    
    # Test temporal shift
    X_shift = temporal_shift(X_test, shift=2)
    print(f"✓ Temporal shift: {X_shift.shape}")
    assert np.array_equal(X_shift[2:], X_test[:-2]), "Time shift failed"
    
    # Test combined augmentation
    config = get_priority1_config()
    X_aug, y_aug = augment_patch_sequence(X_test, y_test, config)
    print(f"✓ Combined augmentation: {X_aug.shape}")
    assert y_aug == y_test, "Label changed by augmentation"
    
    print("\n✓ All augmentation tests passed!")
    print(f"\nPriority 1 config: {config}")
    print(f"Expected augmentation factor: ~24x")
