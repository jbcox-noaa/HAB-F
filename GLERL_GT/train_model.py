import os
import logging
import hashlib

import numpy      as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, Model, Input, optimizers

logging.basicConfig(level=logging.INFO)

channels_dict = {
    "PACE": 172
}

def train_model(
        save_dir: str = './',
        sensor: str = "PACE",
        patch_size: int = 3,
        pm_threshold: float = 0.1
    ) -> tuple:
    """
    Trains a CNN on the provided balanced training data and returns
    (loss, accuracy, auc, f1_score) evaluated on the test set.
    """
    # ─── LOAD DATA ─────────────────────────────────────────
    fn = os.path.join(save_dir, f'training_data_balanced_{sensor}.npy')
    raw = np.load(fn, allow_pickle=True)
    logging.info(f"Loaded {len(raw)} samples from {fn}")

    # Determine feature sizes
    first_flat = raw[0][3]
    n_features = first_flat.size
    feat_patch = patch_size * patch_size * channels_dict[sensor]
    ctx_size = n_features - feat_patch
    if ctx_size <= 0:
        raise ValueError("Context size <=0; check pipeline.")

    # Build arrays
    n = len(raw)
    X_all = np.zeros((n, n_features), dtype=float)
    y_all = np.zeros((n,), dtype=float)
    for i, (_, _, labels, flat, *_) in enumerate(raw):
        X_all[i] = flat
        part = labels[4] if len(labels) >= 5 and not np.isnan(labels[4]) else 0.01
        y_all[i] = part

    # Split patch vs context
    Xp_flat = X_all[:, :feat_patch]
    Xc = X_all[:, feat_patch:]
    Xp = Xp_flat.reshape((n, patch_size, patch_size, channels_dict[sensor]))

    # Add mask channel
    mask = np.any(~np.isnan(Xp), axis=-1, keepdims=True).astype('float32')
    Xp = np.concatenate([Xp, mask], axis=-1)

    # Normalize patch
    means = np.nanmean(Xp[..., :channels_dict[sensor]], axis=(0,1,2))
    stds  = np.nanstd( Xp[..., :channels_dict[sensor]], axis=(0,1,2))
    Xp[..., :channels_dict[sensor]] = (Xp[..., :channels_dict[sensor]] - means) / (stds + 1e-6)
    Xp = np.nan_to_num(Xp, nan=0.0)

    # Normalize context
    ctx_mean = np.nanmean(Xc, axis=0)
    ctx_std  = np.nanstd( Xc, axis=0)
    Xc = (Xc - ctx_mean) / (ctx_std + 1e-6)
    Xc = np.nan_to_num(Xc, nan=0.0)

    # Save normalization stats
    np.save(os.path.join(save_dir, 'context_means.npy'), ctx_mean)
    np.save(os.path.join(save_dir, 'context_stds.npy'),  ctx_std)
    os.makedirs(os.path.join(save_dir, 'channel_stats'), exist_ok=True)
    np.save(os.path.join(save_dir, 'channel_stats', 'means.npy'), means)
    np.save(os.path.join(save_dir, 'channel_stats', 'stds.npy'),  stds)

    # Augment flips
    if patch_size > 1:
        flips = [np.flip(Xp, axis=2), np.flip(Xp, axis=1), np.flip(np.flip(Xp, axis=1), axis=2)]
        Xp = np.concatenate([Xp] + flips, axis=0)
        Xc = np.concatenate([Xc]*4, axis=0)
        y_all = np.concatenate([y_all]*4, axis=0)

    # Split sets
    y_bin = (y_all >= pm_threshold).astype('float32')
    Xp_tmp, Xp_val, Xc_tmp, Xc_val, y_tmp, y_val = train_test_split(
        Xp, Xc, y_bin, test_size=1/8, random_state=42
    )
    Xp_train, Xp_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
        Xp_tmp, Xc_tmp, y_tmp, test_size=1/7, random_state=42
    )

    # ─── MODEL ─────────────────────────────────────────────
    inp_p = Input(shape=(patch_size, patch_size, channels_dict[sensor]+1))
    inp_c = Input(shape=(ctx_size,))

    x = layers.Conv2D(32,(1,1),activation='relu',padding='same')(inp_p)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,(1,1),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((patch_size*patch_size,64))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalMaxPooling1D()(x)

    c = layers.Dense(64, activation='relu')(inp_c)
    c = layers.Dropout(0.1)(c)

    m = layers.Concatenate()([x,c])
    for units in [64,32,4]:
        m = layers.Dense(units, activation='relu')(m)
        m = layers.BatchNormalization()(m)
        m = layers.Dropout(0.1)(m)
    out = layers.Dense(1, activation='sigmoid')(m)

    model = Model([inp_p, inp_c], out)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    chk = ModelCheckpoint(
        filepath=os.path.join(save_dir,'model.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )

    history = model.fit(
        [Xp_train, Xc_train], y_train,
        validation_data=([Xp_val, Xc_val], y_val),
        epochs=300, batch_size=64, verbose=2,
        callbacks=[chk]
    )

    loss, acc, auc = model.evaluate([Xp_test, Xc_test], y_test, verbose=2)
    y_pred = (model.predict([Xp_test, Xc_test], verbose=0).ravel() >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    logging.info(f"Test → loss={loss:.3f}, acc={acc:.3f}, auc={auc:.3f}, f1={f1:.3f}")

    return loss, acc, auc, f1