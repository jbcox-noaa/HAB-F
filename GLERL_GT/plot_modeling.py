#!/usr/bin/env python3
"""
cnn_model_plotter.py

Module to generate daily microcystin‐prediction maps by averaging
across multiple CNN models of potentially different patch sizes.

Usage:
    from cnn_model_plotter import plot_model_results
    plot_model_results(
        save_dir='.',
        dates_to_plot=[...],        # optional list of datetime.date
        prob_threshold=0.5,         # binary threshold
        min_valid_frac=0.5          # minimum valid pixels per patch
    )
"""

import glob
import os
import logging
from datetime import datetime, date, timedelta, timezone

import earthaccess
import xarray            as xr
import numpy             as np
import tensorflow        as tf
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import matplotlib.pyplot as plt

from tqdm import tqdm
from shapely.ops import unary_union
from shapely.vectorized import contains as shp_contains
from shapely.geometry.base import BaseGeometry

from helpers import (
    process_pace_granule,
    extract_pace_patch,
    extract_datetime_from_filename,
    with_retries
)


def plot_model_results(
        save_dir        = './Model_Average4',
        dates_to_plot   = None,
        bbox = (-83.5, 41.3, -81, 42.75),
        prob_threshold  = 0.5,
        min_valid_frac  = 0.5,
        show_rgb = False
    ):
    """
    Generate daily microcystin‐prediction maps for each date in dates_to_plot,
    averaging binary predictions across all CNN models in save_dir matching
    "model*.keras", even if they expect different patch sizes.

    The overlay is solid red with alpha equal to the fraction of models
    predicting the positive class at each pixel.
    """

    # ─── Configuration ────────────────────────────────────────────────────────────
    start_date    = date(2024, 5, 1)
    end_date      = date(2025, 7, 29)
    days_lookback = 7   # look back days_lookback → (days_lookback+1)-day window

    if dates_to_plot is None:
        span = (end_date - start_date).days
        dates_to_plot = [start_date + timedelta(days=i) for i in range(span + 1)]

    out_dir = os.path.join(save_dir, "Images", "Daily_Plots")
    os.makedirs(out_dir, exist_ok=True)

    # ─── Discover & load all CNN models ───────────────────────────────────────────
    logging.info("Discovering CNN models in %s …", save_dir)
    model_paths = sorted(glob.glob(os.path.join(save_dir, "model*.keras")))
    if not model_paths:
        raise FileNotFoundError(f"No models matching 'model*.keras' in {save_dir}")
    models = []
    for mp in model_paths:
        logging.info("  Loading model %s", mp)
        models.append(tf.keras.models.load_model(mp))
    n_models = len(models)
    logging.info("Loaded %d models.", n_models)

    # ─── Load normalization stats & sensor wavelengths ───────────────────────────
    patch_means = np.load(os.path.join('data', 'channel_means.npy'))
    patch_stds  = np.load(os.path.join('data', 'channel_stds.npy'))
    ctx_means   = np.load(os.path.join(save_dir, 'context_means.npy'))
    ctx_stds    = np.load(os.path.join(save_dir, 'context_stds.npy'))

    wl_ref_file = os.path.join('data', 'ref',
                               'PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc')
    wave_all    = (
        xr.open_dataset(wl_ref_file, group="sensor_band_parameters")[
            "wavelength_3d"
        ].data
    )
    dpi  = 100

    # ─── Loop over dates ─────────────────────────────────────────────────────────
    for current_date in dates_to_plot:
        logging.info("Processing date %s …", current_date.isoformat())

        # define temporal window
        t0 = datetime.combine(current_date, datetime.min.time(), timezone.utc) \
             - timedelta(days=days_lookback)
        t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)

        # search & download granules
        results = (
            with_retries(earthaccess.search_data,
                         short_name="PACE_OCI_L2_AOP",
                         temporal=(t0, t1),
                         bounding_box=bbox)
            + with_retries(earthaccess.search_data,
                           short_name="PACE_OCI_L2_AOP_NRT",
                           temporal=(t0, t1),
                           bounding_box=bbox)
        )
        if not results:
            logging.warning("  No granules found; skipping.")
            continue
        files = with_retries(earthaccess.download, results, './data/')
        files_sorted = sorted(
            files,
            key=lambda f: extract_datetime_from_filename(f),
            reverse=True
        )

        # placeholders for lat/lon, rgb composite, masks, vote accumulators
        lat_c = lon_c = None
        rgb_norm = None
        lake_mask = None

        vote_sum   = None  # sum of positive votes at each pixel
        vote_count = None  # number of models that attempted that pixel

        # ─── Iterate granules newest→oldest ────────────────────────────────────
        for fpath in tqdm(files_sorted,
                          desc=f"  Granules for {current_date}",
                          unit="granule"):
            try:
                wls_curr, arr_stack, lat_c_curr, lon_c_curr = process_pace_granule(
                    fpath, bbox, {"res_km":1.2}, wave_all
                )
            except Exception as e:
                logging.warning("    Skipping %s: %s",
                                os.path.basename(fpath), e)
                continue
            if arr_stack is None:
                continue

            ny, nx = arr_stack.shape[1], arr_stack.shape[2]

            # build or update RGB composite
            r = arr_stack[105].T; g = arr_stack[75].T; b = arr_stack[48].T
            rgb_raw = np.dstack((r, g, b)).astype(float)
            mn, mx = np.nanmin(rgb_raw), np.nanmax(rgb_raw)
            if mx <= mn:
                continue
            rgb_curr = (rgb_raw - mn) / (mx - mn)

            if vote_sum is None:
                # first granule → initialize grids and masks
                lat_c, lon_c = lat_c_curr, lon_c_curr
                vote_sum   = np.zeros((ny, nx), dtype=float)
                vote_count = np.zeros((ny, nx), dtype=float)
                rgb_norm   = rgb_curr.copy()
                # lake mask
                lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
                raw_geoms = list(cfeature.LAKES.with_scale('10m').geometries())
                union = unary_union([g for g in raw_geoms
                                     if isinstance(g, BaseGeometry)])
                lake_mask = shp_contains(union, lon_grid, lat_grid)
            else:
                # fill holes in rgb_norm
                holes = np.isnan(rgb_norm[...,0])
                if holes.any():
                    for c in range(3):
                        band = rgb_norm[...,c]
                        band[holes] = rgb_curr[...,c][holes]
                        rgb_norm[...,c] = band

            # normalize context for this granule
            gran_means = np.nanmean(arr_stack, axis=(1,2))
            ctx_norm   = (gran_means - ctx_means) / (ctx_stds + 1e-6)

            # ─── Per-model patch extraction & voting ────────────────────────
            for model in models:
                # extract expected patch size from model.input_shape
                _, Ps, _, _ = model.input_shape[0]
                patch_size_m = int(Ps)

                Xp_m = []
                coords_m = []
                # slide over each pixel
                for i in range(ny):
                    for j in range(nx):
                        pdict = extract_pace_patch(
                            arr_stack, wls_curr,
                            lon_c[j], lat_c[i],
                            patch_size_m, lat_c, lon_c
                        )
                        total = sum(a.size for a in pdict.values())
                        valid = sum(np.count_nonzero(~np.isnan(a))
                                    for a in pdict.values())
                        if total == 0 or (valid/total) < min_valid_frac:
                            continue

                        bs = np.stack([pdict[wl] for wl in wls_curr],
                                      axis=-1)
                        mask = np.any(~np.isnan(bs), axis=-1,
                                      keepdims=True).astype('float32')
                        p4 = np.concatenate([bs, mask], axis=-1)

                        # normalize spectral
                        C = patch_means.shape[0]
                        spec = (p4[...,:C] - patch_means.reshape(1,1,C)) \
                             / (patch_stds.reshape(1,1,C) + 1e-6)
                        p4 = np.concatenate([spec, p4[...,C:]], axis=-1)
                        p4[...,:C] = np.nan_to_num(p4[...,:C], nan=0.0)

                        Xp_m.append(p4)
                        coords_m.append((i,j))

                if not Xp_m:
                    continue

                Xp_m = np.stack(Xp_m, axis=0)
                Xc_m = np.tile(ctx_norm, (len(Xp_m), 1))

                # predict and binarize
                probs = model.predict([Xp_m, Xc_m], verbose=0).ravel()
                bins  = (probs >= prob_threshold).astype(float)

                # accumulate votes & counts
                for (i,j), b in zip(coords_m, bins):
                    vote_sum[i,j]   += b
                    vote_count[i,j] += 1

            # if every pixel has at least one vote, we can stop
            if np.all(vote_count > 0):
                break

        # skip if no valid pixels
        if vote_sum is None or np.all(vote_count == 0):
            logging.warning("  No valid data for %s; skipping.",
                            current_date.isoformat())
            continue

        # fraction positive per pixel
        frac_map = np.full_like(vote_sum, np.nan)
        valid = (vote_count > 0)
        frac_map[valid] = vote_sum[valid] / vote_count[valid]

        # prepare for plotting
        rgb_plot = np.transpose(rgb_norm, (1,0,2))
        mask = ~lake_mask
        for c in range(3):
            rgb_plot[...,c][mask] = np.nan
        frac_plot = frac_map.copy()
        frac_plot[mask] = np.nan

        # plot background + red overlay
        fig, ax = plt.subplots(
            figsize=(8,6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_facecolor("black")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white")
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="black", zorder=0)
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        if show_rgb:
            ax.imshow(rgb_plot, origin="lower", extent=extent,
                    transform=ccrs.PlateCarree(), zorder=2)

        rgba = np.zeros((*frac_plot.shape, 4), dtype=float)
        rgba[...,0] = 1.0        # red channel
        rgba[...,3] = frac_plot  # alpha = fraction positive
        ax.imshow(rgba, origin="lower", extent=extent,
                  transform=ccrs.PlateCarree(), zorder=10)

        ax.coastlines(resolution="10m")
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
        gl.top_labels = gl.right_labels = False

        plt.tight_layout()
        outpath = os.path.join(out_dir, current_date.strftime("%Y%m%d") + ".png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info("  Saved plot → %s", outpath)

    logging.info("All requested dates processed—done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_model_results()
