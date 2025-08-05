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

def plot_model_results(
        patch_size     = 3,
        save_dir       = './',
        dates_to_plot  = None
    ):
    """
    Generate daily microcystin‐probability plots for a specified list of dates.
    If dates_to_plot is None, defaults to every day from start_date to end_date.
    """

    # ─── Configuration ────────────────────────────────────────────────────────────
    start_date    = date(2024, 4, 15)
    end_date      = date(2024, 10, 15)
    days_lookback = 7   # look back 7 days → 8-day window total

    # Build the list of dates if none provided
    if dates_to_plot is None:
        span = (end_date - start_date).days
        dates_to_plot = [ start_date + timedelta(days=i) for i in range(span + 1) ]

    # Output directory for PNGs
    out_dir = os.path.join(save_dir, "Images", "Daily_Plots")
    os.makedirs(out_dir, exist_ok=True)

    # Model + normalization stats + wavelengths file
    model_filepath = os.path.join(save_dir, 'model.keras')
    means_filepath = os.path.join('data', 'channel_means.npy')
    stds_filepath  = os.path.join('data', 'channel_stds.npy')
    wl_ref_file    = os.path.join('data', 'ref',
                                  'PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc')

    # Geographic bounding box (west, south, east, north)
    bbox = (-83.5, 41.3, -82.45, 42.2)
    dpi  = 100

    # ─── Load model and stats ─────────────────────────────────────────────────────
    logging.info("Loading CNN model and normalization stats")
    cnn   = tf.keras.models.load_model(model_filepath)
    means = np.load(means_filepath)
    stds  = np.load(stds_filepath)

    logging.info("Loading sensor wavelengths")
    wave_all = (
        xr.open_dataset(wl_ref_file, group="sensor_band_parameters")
          ["wavelength_3d"]
          .data
    )

    # ─── Loop over only the requested dates ──────────────────────────────────────
    for current_date in dates_to_plot:
        logging.info(f"Building plot for {current_date.isoformat()}")

        # Time window: [current_date – days_lookback, current_date]
        t0 = datetime.combine(current_date, datetime.min.time(), timezone.utc) - \
             timedelta(days=days_lookback)
        t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)
        logging.debug(f"  Searching granules from {t0} to {t1}")

        # Search & download granules (NRT + standard)
        results = (
            with_retries(
                earthaccess.search_data,
                short_name   = "PACE_OCI_L2_AOP",
                temporal     = (t0, t1),
                bounding_box = bbox
            )
            + with_retries(
                earthaccess.search_data,
                short_name   = "PACE_OCI_L2_AOP_NRT",
                temporal     = (t0, t1),
                bounding_box = bbox
            )
        )
        if not results:
            logging.warning("  No granules found; skipping date.")
            continue

        files = with_retries(
            earthaccess.download,
            results,
            './data/'
        )
        # Sort newest → oldest by acquisition time
        files_sorted = sorted(
            files,
            key=lambda f: extract_datetime_from_filename(f),
            reverse=True
        )

        # ─── Hole-filling instead of averaging ──────────────────────────────────
        pacearray = None
        wls = lat_c = lon_c = None

        for fpath in files_sorted:
            try:
                res = process_pace_granule(fpath, bbox, {"res_km":1.2}, wave_all)
            except Exception as e:
                logging.warning(f"    Skipping {os.path.basename(fpath)}: {e}")
                continue
            if not res:
                continue

            wls_curr, arr_stack, lat_c, lon_c = res

            if pacearray is None:
                # initialize from newest granule
                wls = wls_curr
                lat_c_curr, lon_c_curr = lat_c, lon_c
                pacearray = arr_stack.copy()
            else:
                # fill only where pacearray is NaN
                mask_new  = ~np.isnan(arr_stack)
                fill_locs = mask_new & np.isnan(pacearray)
                pacearray[fill_locs] = arr_stack[fill_locs]

            # break early if no holes remain
            if not np.isnan(pacearray).any():
                break

        if pacearray is None or np.isnan(pacearray).all():
            logging.warning("  No valid data to build map; skipping date.")
            continue

        # ─── Build RGB background ───────────────────────────────────────────────
        r_idx, g_idx, b_idx = 105, 75, 48
        r = pacearray[r_idx].T
        g = pacearray[g_idx].T
        b = pacearray[b_idx].T
        rgb = np.dstack((r, g, b))
        rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
        rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)

        # ─── Extract patches & run CNN ─────────────────────────────────────────
        patches, coords = [], []
        for i, la in enumerate(lat_c):
            for j, lo in enumerate(lon_c):
                pd = extract_pace_patch(pacearray, wls, lo, la,
                                        patch_size, lat_c, lon_c)
                bs   = np.stack([pd[wl] for wl in wls], axis=-1)
                mask = np.any(~np.isnan(bs), axis=-1, keepdims=True).astype('float32')
                if mask.sum() == 0:
                    continue
                p4 = np.concatenate([bs, mask], axis=-1)
                p4 = (p4 - means.reshape(1,1,-1)) / (stds.reshape(1,1,-1) + 1e-6)
                patches.append(np.nan_to_num(p4, nan=0.0))
                coords.append((i,j))

        X      = np.stack(patches, axis=0)
        probs  = cnn.predict(X).ravel()
        mc_map = np.full((len(lat_c), len(lon_c)), np.nan)
        for (i,j), p in zip(coords, probs):
            mc_map[i,j] = p

        # ─── Create Lake Mask ───────────────────────────────────────────────────────
        # build one union of all lake geometries at 10m resolution
        from shapely.ops import unary_union
        from shapely.vectorized import contains as shp_contains

        raw_geoms = list(cfeature.LAKES.with_scale('10m').geometries())
        from shapely.geometry.base import BaseGeometry
        lake_geoms = [g for g in raw_geoms if isinstance(g, BaseGeometry)]
        if not lake_geoms:
            logging.error("No valid lake geometries found!")
            return None
        lake_union = unary_union(lake_geoms)

        # create meshgrid of cell‐center coords
        raw_geoms = cfeature.LAKES.with_scale('10m').geometries()
        lake_geoms = [g for g in raw_geoms if isinstance(g, BaseGeometry)]
        lake_union = unary_union(lake_geoms)

        # 3) build grid & vectorized mask
        lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
        lake_mask = shp_contains(lake_union, lon_grid, lat_grid)

        # apply mask: only show lake pixels
        rgb_norm.transpose(1,0,2)[~lake_mask] = np.nan
        mc_map[~lake_mask]   = np.nan

        # ─── Plot & save ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(
            figsize=(8,6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_facecolor("black")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white")
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="black", zorder=0)

        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        ax.imshow(
            rgb_norm.transpose(1,0,2),
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
            zorder=2,
            interpolation="nearest"
        )

        # overlay pure-red where prob ≥ 0.5
        show_mask = (~np.isnan(mc_map)) & (mc_map >= 0.5)
        rgba      = np.zeros((mc_map.shape[0], mc_map.shape[1], 4), dtype=float)
        rgba[...,0][show_mask] = 1.0   # red
        rgba[...,3][show_mask] = 1.0   # alpha
        ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            zorder=10
        )

        ax.coastlines(resolution="10m")
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # colorbar (optional)
        m = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0,1))
        m.set_array(mc_map)
        cbar = plt.colorbar(m, ax=ax, orientation="vertical",
                             pad=0.02, shrink=0.7)
        cbar.set_label("Microcystin Probability")

        plt.tight_layout()
        outpath = os.path.join(out_dir, current_date.strftime("%Y%m%d") + ".png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"  Saved plot → {outpath}")

        # ─── Cleanup only files older than current_date – days_lookback ───────────
        cutoff = datetime.combine(current_date, datetime.min.time(), timezone.utc) \
                 - timedelta(days=days_lookback)
        for fname in os.listdir('data'):
            if not fname.endswith('.nc'):
                continue
            fpath = os.path.join('data', fname)
            try:
                fdate = extract_datetime_from_filename(fpath)
                fdate = fdate.replace(tzinfo=timezone.utc)
                
                if fdate < cutoff:
                    #os.remove(fpath)
                    logging.debug(f"  Removed old granule: {fname}")
            except Exception as e:
                logging.warning(f"  Could not parse date for {fname}: {e}")

    logging.info("All requested dates processed—done!")
