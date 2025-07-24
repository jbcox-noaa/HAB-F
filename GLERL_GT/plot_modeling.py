import glob
import os
import logging
import earthaccess

import xarray            as xr
import numpy             as np
import tensorflow        as tf
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import matplotlib.pyplot as plt

from datetime import datetime, date, timedelta, timezone
from helpers import (
    process_pace_granule,
    extract_pace_patch,
    extract_datetime_from_filename,
    with_retries
)
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.vectorized import contains as shp_contains
from shapely.geometry.base import BaseGeometry


def plot_model_results(
        patch_size      = 3,
        save_dir        = './',
        dates_to_plot   = None,
        prob_threshold  = 0.5,
        min_valid_frac  = 0.5
    ):
    """
    Generate daily microcystin‐prediction maps by averaging across all
    models in `save_dir` whose filenames match "model*.keras".
    The overlay is red; alpha = fraction_of_models_predicting_positive.
    """

    # ─── Configuration ────────────────────────────────────────────────────────────
    start_date    = date(2025, 6, 1)
    end_date      = date(2025, 7, 21)
    days_lookback = 7

    if dates_to_plot is None:
        span = (end_date - start_date).days
        dates_to_plot = [start_date + timedelta(days=i) for i in range(span + 1)]

    out_dir = os.path.join(save_dir, "Images", "Daily_Plots")
    os.makedirs(out_dir, exist_ok=True)

    # ─── Load all models ──────────────────────────────────────────────────────────
    logging.info("Loading all CNN models from %s", save_dir)
    model_paths = sorted(glob.glob(os.path.join(save_dir, "model*.keras")))
    if not model_paths:
        raise FileNotFoundError(f"No models matching 'model*.keras' in {save_dir}")
    models = [tf.keras.models.load_model(mp) for mp in model_paths]
    n_models = len(models)
    logging.info("  Loaded %d models: %s", n_models, model_paths)

    # ─── Load stats & wavelengths ─────────────────────────────────────────────────
    patch_means = np.load(os.path.join(save_dir, 'data/channel_means.npy'))
    patch_stds  = np.load(os.path.join(save_dir, 'data/channel_stds.npy'))
    ctx_means   = np.load(os.path.join(save_dir, 'context_means.npy'))
    ctx_stds    = np.load(os.path.join(save_dir, 'context_stds.npy'))
    wl_ref_file = os.path.join('data','ref','PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc')
    wave_all    = (
        xr.open_dataset(wl_ref_file, group="sensor_band_parameters")
          ["wavelength_3d"].data
    )

    bbox = (-83.5, 41.3, -82.45, 42.2)
    dpi  = 100

    # ─── Process each date ────────────────────────────────────────────────────────
    for current_date in dates_to_plot:
        logging.info(f"Building plot for {current_date.isoformat()}")
        t0 = datetime.combine(current_date, datetime.min.time(), timezone.utc) - timedelta(days=days_lookback)
        t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)

        results = (
            with_retries(earthaccess.search_data,
                         short_name="PACE_OCI_L2_AOP",
                         temporal=(t0, t1), bounding_box=bbox)
            + with_retries(earthaccess.search_data,
                           short_name="PACE_OCI_L2_AOP_NRT",
                           temporal=(t0, t1), bounding_box=bbox)
        )
        if not results:
            logging.warning("  No granules found; skipping date.")
            continue

        files = with_retries(earthaccess.download, results, './data/')
        files_sorted = sorted(files,
                              key=lambda f: extract_datetime_from_filename(f),
                              reverse=True)

        # placeholders
        lat_c = lon_c = None
        mc_map = None
        rgb_norm = None

        # lake mask
        lake_mask = None

        for fpath in tqdm(files_sorted, desc=f"  Granules for {current_date}", unit="granule"):
            try:
                wls_curr, arr_stack, lat_c_curr, lon_c_curr = process_pace_granule(
                    fpath, bbox, {"res_km":1.2}, wave_all
                )
            except Exception as e:
                logging.warning(f"    Skipping {os.path.basename(fpath)}: {e}")
                continue
            if arr_stack is None:
                continue

            ny, nx = arr_stack.shape[1], arr_stack.shape[2]
            # build RGB
            r = arr_stack[105].T; g = arr_stack[75].T; b = arr_stack[48].T
            rgb_raw = np.dstack((r, g, b)).astype(float)
            curr_min, curr_max = np.nanmin(rgb_raw), np.nanmax(rgb_raw)
            if curr_max <= curr_min:
                continue
            rgb_curr = (rgb_raw - curr_min) / (curr_max - curr_min)

            if mc_map is None:
                # first granule → initialize all
                lat_c, lon_c = lat_c_curr, lon_c_curr
                mc_map = np.full((ny, nx), np.nan, dtype=float)
                rgb_norm = rgb_curr.copy()
                lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
                raw_geoms = list(cfeature.LAKES.with_scale('10m').geometries())
                lake_union = unary_union([g for g in raw_geoms if isinstance(g, BaseGeometry)])
                lake_mask = shp_contains(lake_union, lon_grid, lat_grid)
            else:
                # fill holes in rgb_norm
                holes = np.isnan(rgb_norm[...,0])
                if holes.any():
                    for c in range(3):
                        band = rgb_norm[...,c]
                        band[holes] = rgb_curr[...,c][holes]
                        rgb_norm[...,c] = band

            # compute context once
            gran_means = np.nanmean(arr_stack, axis=(1,2))
            ctx_norm   = (gran_means - ctx_means) / (ctx_stds + 1e-6)

            # build patches
            X_patches, X_ctx, coords = [], [], []
            for i in range(ny):
                for j in range(nx):
                    pdict = extract_pace_patch(arr_stack, wls_curr,
                                               lon_c[j], lat_c[i],
                                               patch_size, lat_c, lon_c)
                    total = sum(a.size for a in pdict.values())
                    valid = sum(np.count_nonzero(~np.isnan(a)) for a in pdict.values())
                    if total==0 or (valid/total)<min_valid_frac:
                        continue

                    bs = np.stack([pdict[wl] for wl in wls_curr], axis=-1)
                    mask = np.any(~np.isnan(bs), axis=-1, keepdims=True).astype('float32')
                    p4 = np.concatenate([bs, mask], axis=-1)

                    # normalize spectral bands
                    C = patch_means.shape[0]
                    spec = (p4[...,:C] - patch_means.reshape(1,1,C)) \
                         / (patch_stds.reshape(1,1,C) + 1e-6)
                    p4 = np.concatenate([spec, p4[...,C:]], axis=-1)
                    p4[...,:C] = np.nan_to_num(p4[...,:C], nan=0.0)

                    X_patches.append(p4)
                    X_ctx.append(ctx_norm)
                    coords.append((i,j))

            if not X_patches:
                continue

            Xp = np.stack(X_patches, axis=0)
            Xc = np.stack(X_ctx,     axis=0)

            # ─── Multi‑model prediction & averaging ──────────────────────────────
            # collect binary predictions from each model
            bin_preds = []
            for m in models:
                probs = m.predict([Xp, Xc], verbose=0).ravel()
                bin_preds.append((probs >= prob_threshold).astype(float))
            # shape: (n_models, n_patches)
            bin_preds = np.stack(bin_preds, axis=0)

            # take mean across models: yields fraction ∈ [0,1]
            frac_pos = bin_preds.mean(axis=0)

            # fill mc_map with that fraction
            for (i,j), frac in zip(coords, frac_pos):
                if np.isnan(mc_map[i,j]):
                    mc_map[i,j] = frac

            # stop early if fully filled
            if not np.isnan(mc_map).any() and not np.isnan(rgb_norm[...,0]).any():
                break

        if mc_map is None or np.all(np.isnan(mc_map)):
            logging.warning(f"  No valid data for {current_date}; skipping.")
            continue

        # mask and transpose for plotting
        rgb_plot = np.transpose(rgb_norm, (1,0,2))
        mask = ~lake_mask
        for c in range(3):
            rgb_plot[...,c][mask] = np.nan
        mc_plot = mc_map.copy()
        mc_plot[mask] = np.nan

        fig, ax = plt.subplots(
            figsize=(8,6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_facecolor("black")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white")
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="black", zorder=0)

        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        ax.imshow(rgb_plot, origin="lower", extent=extent,
                  transform=ccrs.PlateCarree(), zorder=2)

        # overlay average predictions in red
        rgba = np.zeros((*mc_plot.shape, 4), dtype=float)
        rgba[...,0] = 1.0            # red channel
        rgba[...,3] = mc_plot        # alpha = fraction positive
        ax.imshow(rgba, origin="lower", extent=extent,
                  transform=ccrs.PlateCarree(), zorder=10)

        ax.coastlines(resolution="10m")
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
        gl.top_labels = gl.right_labels = False

        plt.tight_layout()
        outpath = os.path.join(out_dir, current_date.strftime("%Y%m%d") + ".png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"  Saved plot → {outpath}")

    logging.info("All requested dates processed—done!")
