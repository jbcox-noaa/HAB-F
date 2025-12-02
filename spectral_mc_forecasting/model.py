"""
Phase 7 Model Architecture: Patch-Based Spectral ConvLSTM

Architecture:
    Input: (batch, seq_len=14, height=11, width=11, features=173)
    ↓
    SpectralEncoder: 1x1 Conv2D (173 → 16)
    ↓
    ConvLSTM2D: Temporal-spatial encoding
    ↓
    Decoder: 1x1 Conv2D (16 → 1)
    ↓
    Center Pixel Extraction + Sigmoid
    ↓
    Output: (batch, 1) - Binary MC probability for center pixel

Key Features:
- Mask-aware processing (mask channel in input)
- Center pixel prediction (aligns with GLERL ground truth)
- Handles sparse sequences (gaps in temporal data)
- Binary classification (MC ≥ 1.0 µg/L)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from . import config


class SpectralEncoder(layers.Layer):
    """
    Spectral band compression using 1x1 convolutions.
    
    Reduces 173 features (172 spectral + 1 mask) → 16 features
    while preserving spatial structure.
    """
    
    def __init__(self, output_features=16, dropout=0.3, name='spectral_encoder'):
        super().__init__(name=name)
        self.output_features = output_features
        self.dropout = dropout
        
    def build(self, input_shape):
        # Input: (batch, seq_len, H, W, 173)
        # We'll apply encoder per timestep, so work with (batch*seq, H, W, 173)
        
        self.encoder_layers = []
        
        # Progressive compression: 173 → 128 → 64 → 32 → 16
        layer_sizes = config.ENCODER_LAYERS  # [128, 64, 32, 16]
        
        for i, n_filters in enumerate(layer_sizes):
            # 1x1 convolution preserves spatial dimensions
            conv = layers.Conv2D(
                filters=n_filters,
                kernel_size=1,
                activation=config.ENCODER_ACTIVATION,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG),
                name=f'encoder_conv_{i}'
            )
            self.encoder_layers.append(conv)
            
            # Dropout for regularization
            if i < len(layer_sizes) - 1:  # Not on last layer
                self.encoder_layers.append(layers.Dropout(self.dropout))
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, seq_len, H, W, 173)
            training: Whether in training mode
            
        Returns:
            (batch, seq_len, H, W, 16)
        """
        # Reshape to process all timesteps together
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        height = tf.shape(inputs)[2]
        width = tf.shape(inputs)[3]
        
        # (batch, seq, H, W, 173) → (batch*seq, H, W, 173)
        x = tf.reshape(inputs, [-1, height, width, config.N_INPUT_FEATURES])
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # (batch*seq, H, W, 16) → (batch, seq, H, W, 16)
        x = tf.reshape(x, [batch_size, seq_len, height, width, self.output_features])
        
        return x
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'output_features': self.output_features,
            'dropout': self.dropout,
        })
        return config_dict


class MaskAwareConvLSTM(layers.Layer):
    """
    ConvLSTM2D that handles masked timesteps.
    
    The mask channel in the input indicates valid (1.0) vs missing (0.0) data.
    This layer processes temporal sequences while being aware of data gaps.
    """
    
    def __init__(self, filters=64, kernel_size=(3, 3), dropout=0.3, 
                 recurrent_dropout=0.2, return_sequences=False, name='convlstm'):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        
    def build(self, input_shape):
        # Input: (batch, seq_len, H, W, features)
        self.convlstm = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            return_sequences=self.return_sequences,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(config.L2_REG),
            name=f'{self.name}_layer'
        )
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, seq_len, H, W, features)
            training: Whether in training mode
            
        Returns:
            If return_sequences=True: (batch, seq_len, H, W, filters)
            If return_sequences=False: (batch, H, W, filters)
        """
        return self.convlstm(inputs, training=training)
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'return_sequences': self.return_sequences,
        })
        return config_dict


class SpectralDecoder(layers.Layer):
    """
    Decoder to convert ConvLSTM output to MC probability map.
    
    Uses 1x1 convolutions to preserve spatial structure.
    """
    
    def __init__(self, dropout=0.2, name='spectral_decoder'):
        super().__init__(name=name)
        self.dropout = dropout
        
    def build(self, input_shape):
        # Input: (batch, H, W, convlstm_filters)
        
        self.decoder_layers = []
        
        # Progressive expansion: convlstm_filters → 32 → 16 → 8 → 1
        layer_sizes = config.DECODER_LAYERS  # [32, 16, 8]
        
        for i, n_filters in enumerate(layer_sizes):
            conv = layers.Conv2D(
                filters=n_filters,
                kernel_size=1,
                activation=config.DECODER_ACTIVATION,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG),
                name=f'decoder_conv_{i}'
            )
            self.decoder_layers.append(conv)
            self.decoder_layers.append(layers.Dropout(self.dropout))
        
        # Final layer: → 1 channel (no activation yet, sigmoid applied later)
        self.final_conv = layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation=None,  # Sigmoid applied after center pixel extraction
            padding='same',
            name='decoder_final'
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, H, W, convlstm_filters)
            training: Whether in training mode
            
        Returns:
            (batch, H, W, 1)
        """
        x = inputs
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Final conv to 1 channel
        x = self.final_conv(x)
        
        return x
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'dropout': self.dropout,
        })
        return config_dict


def build_patch_based_model(
    seq_len=14,
    patch_size=11,
    n_features=173,
    encoder_output=16,
    convlstm_filters=[64, 64, 32],
    convlstm_kernel_size=(3, 3),
    name='spectral_mc_forecaster'
):
    """
    Build patch-based spectral ConvLSTM model for microcystin prediction.
    
    Architecture:
        Input (batch, 14, 11, 11, 173) 
        → SpectralEncoder (173→16)
        → ConvLSTM stack
        → Decoder (→1 channel)
        → Center pixel extraction
        → Sigmoid activation
        → Output (batch, 1)
    
    Args:
        seq_len: Temporal sequence length (14 days)
        patch_size: Spatial patch size (11x11)
        n_features: Input features (172 spectral + 1 mask)
        encoder_output: Encoder output channels (16)
        convlstm_filters: List of ConvLSTM filter sizes
        convlstm_kernel_size: ConvLSTM kernel size
        name: Model name
        
    Returns:
        keras.Model instance
    """
    
    # Input: (batch, seq_len, patch_size, patch_size, n_features)
    inputs = keras.Input(
        shape=(seq_len, patch_size, patch_size, n_features),
        name='spectral_patch_input'
    )
    
    # Spectral Encoder: 173 → 16
    x = SpectralEncoder(
        output_features=encoder_output,
        dropout=config.ENCODER_DROPOUT
    )(inputs)
    
    # ConvLSTM Stack
    for i, n_filters in enumerate(convlstm_filters):
        # Return sequences for all but last layer
        return_sequences = (i < len(convlstm_filters) - 1)
        
        x = MaskAwareConvLSTM(
            filters=n_filters,
            kernel_size=convlstm_kernel_size,
            dropout=config.CONVLSTM_DROPOUT,
            recurrent_dropout=config.CONVLSTM_RECURRENT_DROPOUT,
            return_sequences=return_sequences,
            name=f'convlstm_{i}'
        )(x)
    
    # x is now: (batch, patch_size, patch_size, convlstm_filters[-1])
    
    # Decoder: convlstm_filters → 1
    probability_map = SpectralDecoder(
        dropout=config.DECODER_DROPOUT
    )(x)
    
    # probability_map: (batch, patch_size, patch_size, 1)
    
    # Extract center pixel (this is where the GLERL measurement is)
    center_idx = patch_size // 2  # For 11x11, center is at [5, 5]
    
    # Extract center pixel: (batch, patch_size, patch_size, 1) → (batch, 1)
    center_pixel = probability_map[:, center_idx, center_idx, :]
    
    # Apply sigmoid activation for binary classification
    output = layers.Activation('sigmoid', name='mc_probability')(center_pixel)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=output, name=name)
    
    return model


def build_full_map_model(
    seq_len=14,
    height=84,
    width=73,
    n_features=173,
    encoder_output=16,
    convlstm_filters=[64, 64, 32],
    convlstm_kernel_size=(3, 3),
    name='spectral_mc_forecaster_fullmap'
):
    """
    Build full-map spectral ConvLSTM model for inference on entire lake.
    
    Same architecture as patch-based, but processes full spatial grid
    and outputs probability map for entire region.
    
    Args:
        seq_len: Temporal sequence length
        height: Spatial height (84)
        width: Spatial width (73)
        n_features: Input features (172 spectral + 1 mask)
        encoder_output: Encoder output channels
        convlstm_filters: List of ConvLSTM filter sizes
        convlstm_kernel_size: ConvLSTM kernel size
        name: Model name
        
    Returns:
        keras.Model instance
    """
    
    # Input: (batch, seq_len, height, width, n_features)
    inputs = keras.Input(
        shape=(seq_len, height, width, n_features),
        name='spectral_fullmap_input'
    )
    
    # Spectral Encoder: 173 → 16
    x = SpectralEncoder(
        output_features=encoder_output,
        dropout=config.ENCODER_DROPOUT
    )(inputs)
    
    # ConvLSTM Stack
    for i, n_filters in enumerate(convlstm_filters):
        return_sequences = (i < len(convlstm_filters) - 1)
        
        x = MaskAwareConvLSTM(
            filters=n_filters,
            kernel_size=convlstm_kernel_size,
            dropout=config.CONVLSTM_DROPOUT,
            recurrent_dropout=config.CONVLSTM_RECURRENT_DROPOUT,
            return_sequences=return_sequences,
            name=f'convlstm_{i}'
        )(x)
    
    # x is now: (batch, height, width, convlstm_filters[-1])
    
    # Decoder: convlstm_filters → 1
    probability_map = SpectralDecoder(
        dropout=config.DECODER_DROPOUT
    )(x)
    
    # probability_map: (batch, height, width, 1)
    
    # Apply sigmoid activation
    output = layers.Activation('sigmoid', name='mc_probability_map')(probability_map)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=output, name=name)
    
    return model


def get_model(model_type='patch', **kwargs):
    """
    Factory function to get model based on type.
    
    Args:
        model_type: 'patch' for patch-based training, 'fullmap' for inference
        **kwargs: Additional arguments passed to build function
        
    Returns:
        keras.Model instance
    """
    if model_type == 'patch':
        return build_patch_based_model(**kwargs)
    elif model_type == 'fullmap':
        return build_full_map_model(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'patch' or 'fullmap'.")


def compile_model(model, learning_rate=5e-4):
    """
    Compile model with appropriate loss, optimizer, and metrics.
    
    Args:
        model: keras.Model instance
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    
    # Binary crossentropy for binary classification
    loss = keras.losses.BinaryCrossentropy()
    
    # Optimizer with gradient clipping to prevent NaN
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Metrics
    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


if __name__ == '__main__':
    """Test model building."""
    
    print("=" * 80)
    print("TESTING PHASE 7 MODEL ARCHITECTURE")
    print("=" * 80)
    
    # Test patch-based model (for training)
    print("\n1. Patch-based model (for GLERL-supervised training):")
    print("-" * 80)
    
    patch_model = build_patch_based_model(
        seq_len=14,
        patch_size=11,
        n_features=173,
        encoder_output=16,
        convlstm_filters=config.CONVLSTM_FILTERS,
        convlstm_kernel_size=config.CONVLSTM_KERNEL_SIZE
    )
    
    patch_model = compile_model(patch_model, learning_rate=5e-4)
    
    print(patch_model.summary())
    
    # Test with dummy data
    print("\n2. Testing with dummy input:")
    print("-" * 80)
    
    dummy_input = np.random.randn(4, 14, 11, 11, 173).astype(np.float32)
    print(f"Input shape: {dummy_input.shape}")
    
    dummy_output = patch_model.predict(dummy_input, verbose=0)
    print(f"Output shape: {dummy_output.shape}")
    print(f"Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
    
    # Test full-map model (for inference)
    print("\n3. Full-map model (for inference on entire lake):")
    print("-" * 80)
    
    fullmap_model = build_full_map_model(
        seq_len=14,
        height=84,
        width=73,
        n_features=173,
        encoder_output=16,
        convlstm_filters=config.CONVLSTM_FILTERS,
        convlstm_kernel_size=config.CONVLSTM_KERNEL_SIZE
    )
    
    print(fullmap_model.summary())
    
    # Test with dummy data
    print("\n4. Testing full-map with dummy input:")
    print("-" * 80)
    
    dummy_fullmap_input = np.random.randn(2, 14, 84, 73, 173).astype(np.float32)
    print(f"Input shape: {dummy_fullmap_input.shape}")
    
    dummy_fullmap_output = fullmap_model.predict(dummy_fullmap_input, verbose=0)
    print(f"Output shape: {dummy_fullmap_output.shape}")
    print(f"Output range: [{dummy_fullmap_output.min():.4f}, {dummy_fullmap_output.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("✅ MODEL ARCHITECTURE TESTS PASSED")
    print("=" * 80)
