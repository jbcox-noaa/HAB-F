import os
import logging
import hashlib

import numpy      as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import Input, Model

logging.basicConfig(level=logging.INFO)

channels_dict = {
    "PACE": 172
}

def train_model(
        save_dir     = './',
        sensor       = "PACE",
        patch_size   = 3,
        pm_threshold = 0.1
    ):

    # ─── LOAD & UNPACK ──────────────────────────────────────────────────────────
    training_data_filename = os.path.join(
        save_dir, f'training_data_balanced_{sensor}.npy')
    raw = np.load(training_data_filename, allow_pickle=True)
    logging.info(f"Loaded {len(raw)} entries from {training_data_filename}")

    # determine the two slices: 
    #   1) the spatial patch   
    #   2) the appended context (global means)
    first_patch = raw[0][3]
    n_features = first_patch.size
    # how many features come from a pure patch?
    feat_per_patch = patch_size * patch_size * channels_dict[sensor]
    # the remainder must be the appended means
    context_size = n_features - feat_per_patch

    if context_size <= 0:
        raise ValueError("No extra context detected—check your pipeline")

    # build X_patch, X_ctx, and y
    n_samples = len(raw)
    X_all    = np.zeros((n_samples, n_features), dtype=float)
    y_all    = np.zeros((n_samples,),       dtype=float)
    for i, (_, _, labels, flat) in enumerate(raw):
        if flat.size != n_features:
            raise ValueError(f"Entry {i}: expected {n_features}, got {flat.size}")
        X_all[i, :] = flat
        # your existing logic to extract y
        part = 0.0
        if labels is not None and len(labels) >= 5:
            part = labels[4] if not np.isnan(labels[4]) else 0.0
            part = part if part != 0 else 0.01
        y_all[i] = part

    # split patch vs. context
    X_patch_flat = X_all[:, :feat_per_patch]
    X_ctx        = X_all[:, feat_per_patch:]
    # reshape the patch back into (H,W,C)
    X_patch = X_patch_flat.reshape(
        (n_samples, patch_size, patch_size, channels_dict[sensor])
    )

    # ─── BUILD THE MASK & CONCAT ───────────────────────────────────────────────
    mask = np.any(~np.isnan(X_patch), axis=-1, keepdims=True).astype('float32')
    X_patch = np.concatenate([X_patch, mask], axis=-1)  # now shape (…, C+1)

    # ─── NORMALIZE ────────────────────────────────────────────────────────────
    # patch branch normalization (as before)
    means = np.nanmean(X_patch[..., :channels_dict[sensor]], axis=(0,1,2))
    stds  = np.nanstd( X_patch[..., :channels_dict[sensor]], axis=(0,1,2))
    Xp = X_patch.copy()
    Xp[..., :channels_dict[sensor]] = (Xp[..., :channels_dict[sensor]] - means) / (stds + 1e-6)
    Xp = np.nan_to_num(Xp, nan = 0.0)

    # context branch normalization
    ctx_mean = np.nanmean(X_ctx, axis=0)
    ctx_std  = np.nanstd( X_ctx, axis=0)
    Xc = (X_ctx - ctx_mean) / (ctx_std + 1e-6)
    Xc = np.nan_to_num(Xc, nan=0.0)

        # after computing ctx_mean and ctx_std (before or after augmentation):
    ctx_means = ctx_mean   # shape = (n_wl,)
    ctx_stds  = ctx_std

    # choose a directory to hold these—e.g. './data'
    np.save(f'{save_dir}/context_means.npy', ctx_means)
    np.save(f'{save_dir}/context_stds.npy',  ctx_stds)

    # you might also save patch‐branch stats if not already
    np.save('data/channel_means.npy', means)
    np.save('data/channel_stds.npy',  stds)

    # ─── AUGMENT (flips) ──────────────────────────────────────────────────────
    if patch_size > 1:
        flip_y  = np.flip(Xp, axis=2)
        flip_x  = np.flip(Xp, axis=1)
        flip_xy = np.flip(flip_x, axis=2)
        Xp = np.concatenate([Xp, flip_y, flip_x, flip_xy], axis=0)
        # replicate contexts & labels
        Xc = np.concatenate([Xc, Xc, Xc, Xc], axis=0)
        y_all = np.concatenate([y_all, y_all, y_all, y_all], axis=0)

    # ─── TRAIN/VAL/TEST SPLIT ─────────────────────────────────────────────────
    y_bin = (y_all >= pm_threshold).astype('float32').reshape(-1,1)
    # stratify if you like; here simple random split
    Xp_tmp, Xp_val, Xc_tmp, Xc_val, y_tmp, y_val = train_test_split(
        Xp,     Xc,     y_bin, test_size=1/8, random_state=42)
    Xp_train, Xp_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
        Xp_tmp, Xc_tmp, y_tmp, test_size=1/7, random_state=42)

    # ─── MODEL DEFINITION (two inputs) ────────────────────────────────────────
    inp_patch = Input(shape=(patch_size, patch_size, channels_dict[sensor] + 1))
    inp_ctx   = Input(shape=(context_size,))

    # CNN branch on the patch
    x = layers.Conv2D(32, (1,1), activation='relu', padding='same')(inp_patch)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (1,1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((patch_size*patch_size, 64))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # optionally map the context through its own small MLP
    c = layers.Dense(64, activation='relu')(inp_ctx)
    c = layers.Dropout(0.1)(c)

    # merge
    m = layers.Concatenate()([x, c])
    m = layers.Dense(64, activation='relu')(m)
    m = layers.BatchNormalization()(m)
    m = layers.Dropout(0.1)(m)
    m = layers.Dense(32, activation='relu')(m)
    m = layers.Dropout(0.1)(m)
    m = layers.Dense(4, activation='relu')(m)
    m = layers.Dropout(0.1)(m)

    out = layers.Dense(1, activation='sigmoid')(m)

    model = Model([inp_patch, inp_ctx], out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # ─── CHECKPOINT ──────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{save_dir}model.keras",
        monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    )

    # ─── TRAIN ───────────────────────────────────────────────────────────────
    model.fit(
        [Xp_train, Xc_train], y_train,
        validation_data=([Xp_val,   Xc_val],   y_val),
        epochs=300, batch_size=64, verbose=2,
        callbacks=[checkpoint_callback]
    )

    # you can evaluate on test set:
    loss, acc, auc = model.evaluate([Xp_test, Xc_test], y_test, verbose=2)
    logging.info(f"test → loss {loss:.3f}, acc {acc:.3f}, auc {auc:.3f}")
