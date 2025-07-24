#!/usr/bin/env python3
"""
cnn_labeling_labeler.py
Standalone Matplotlib+Cartopy GUI for labeling misclassified pixels with interactive labeling.
"""
import sys
import os
import csv
import logging
from datetime import datetime, date, timedelta, timezone

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthaccess

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops            import unary_union
from shapely.vectorized    import contains as shp_contains


# Helpers (ensure these exist in helpers.py)
from helpers import (
    process_pace_granule,
    extract_datetime_from_filename,
    extract_pace_patch,
    with_retries
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
START_DATE     = date(2024, 5, 9)
END_DATE       = date(2024, 10, 15)
LOOKBACK_DAYS  = 7   # 2-day window total
PATCH_SIZE     = 5
BBOX           = (-83.5, 41.3, -82.45, 42.2)  # (west, south, east, north)
CACHE_DIR      = './cache/'
DATA_DIR       = './data/'
CSV_PATH       = './user-labels.csv'
MODEL_PATH     = './Grid_search_oversample3/3day_5px_0.1pm/model.keras'
MEANS_PATH     = os.path.join('data', 'channel_means.npy')
STDS_PATH      = os.path.join('data', 'channel_stds.npy')
WAVELENGTH_REF = os.path.join('data', 'ref', 'PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc')
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-load model & stats
import tensorflow as tf
try:
    cnn = tf.keras.models.load_model(MODEL_PATH)
    means = np.load(MEANS_PATH)
    stds = np.load(STDS_PATH)
    wave_all = (
        xr.open_dataset(WAVELENGTH_REF, group='sensor_band_parameters')['wavelength_3d'].data
    )
    logging.info("Loaded CNN model and stats")
except Exception as e:
    logging.error(f"Error loading model or stats: {e}")
    sys.exit(1)

# Dates list and labels store
dates_to_plot = [
    date(2024, 4, 17),
    date(2024, 4, 20),
    date(2024, 5, 6),
    date(2024, 6, 4),
]

def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

#dates_to_plot = list(daterange(START_DATE, END_DATE))

all_labels = []
idx = 0

# Globals storing the last composite and grid and mask
_last_rgb = None
_last_mc_map = None
_last_lat_c = None
_last_lon_c = None
_last_lake_mask = None

# Utility: save CSV
def save_csv():
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date','lat','lon','label'])
        writer.writeheader()
        writer.writerows(all_labels)
    logging.info(f"Saved {len(all_labels)} records to {CSV_PATH}")

def compute_composite(current_date):
    # ——— caching layer —————————————————————————————————————————
    cache_path = os.path.join(CACHE_DIR, f'{current_date.isoformat()}.npz')
    if os.path.isfile(cache_path):
        with np.load(cache_path) as d:
            logging.info(f"Loaded composite for {current_date} from cache")
            return (
                d['rgb_norm'],
                d['mc_map'],
                d['lat_c'],
                d['lon_c'],
                d['lake_mask']
            )

    # ——— existing code follows ————————————————————————————————————
    t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)
    t0 = t1 - timedelta(days=LOOKBACK_DAYS)
    logging.info(f"Searching granules {t0} to {t1}")
    try:
        r1 = with_retries(earthaccess.search_data,
                           short_name='PACE_OCI_L2_AOP', temporal=(t0,t1), bounding_box=BBOX)
        r2 = with_retries(earthaccess.search_data,
                           short_name='PACE_OCI_L2_AOP_NRT', temporal=(t0,t1), bounding_box=BBOX)
    except Exception as e:
        logging.warning(f"Search failed: {e}")
        return None
    
    results = r1 + r2
    if not results:
        return None
    files = with_retries(earthaccess.download, results, DATA_DIR)
    files_sorted = sorted(files, key=extract_datetime_from_filename, reverse=True)

    lat_c = lon_c = None

    rgb_norm = None
    mc_map   = None
    lake_mask = None
    lat_c = lon_c = None

    for fpath in files_sorted:
        try:
            wls_curr, arr_stack, lat_c_curr, lon_c_curr = \
                process_pace_granule(fpath, BBOX, {'res_km':1.2}, wave_all)
        except Exception:
            continue

        # dimensions
        ny, nx = arr_stack.shape[1], arr_stack.shape[2]

        # 2) per‐granule RGB
        r = arr_stack[105].T
        g = arr_stack[75].T
        b = arr_stack[48].T
        rgb_raw = np.dstack((r, g, b)).astype(float)
        curr_min, curr_max = np.nanmin(rgb_raw), np.nanmax(rgb_raw)
        if curr_max <= curr_min:
            continue
        rgb_curr = (rgb_raw - curr_min) / (curr_max - curr_min)

        # 3) initialize on first good granule
        if rgb_norm is None:
            lat_c, lon_c = lat_c_curr, lon_c_curr
            rgb_norm = rgb_curr.copy()
            mc_map   = np.full((ny, nx), np.nan, dtype=float)

            # build lake mask once
            lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
            raw_geoms = list(cfeature.LAKES.with_scale('10m').geometries())
            lake_union = unary_union([g for g in raw_geoms if isinstance(g, BaseGeometry)])
            lake_mask = shp_contains(lake_union, lon_grid, lat_grid)

        else:
            # 4) fill holes in the existing RGB composite
            holes = np.isnan(rgb_norm[...,0])
            if holes.any():
                for c in range(3):
                    band = rgb_norm[...,c]
                    band[holes] = rgb_curr[...,c][holes]
                    rgb_norm[...,c] = band

        # 5) extract patches, normalize, predict only where mc_map is still NaN
        Xb, coords = [], []
        for i in range(ny):
            for j in range(nx):
                if not np.isnan(mc_map[i,j]):
                    continue   # already filled
                patch = extract_pace_patch(arr_stack, wls_curr,
                                           lon_c[j], lat_c[i],
                                           PATCH_SIZE, lat_c, lon_c)
                # skip if too few valid
                total = sum(arr.size for arr in patch.values())
                valid = sum(np.count_nonzero(~np.isnan(arr)) for arr in patch.values())
                if total==0 or valid/total < 0.5:
                    continue
                bs   = np.stack([patch[wl] for wl in wls_curr], axis=-1)
                mask = np.any(~np.isnan(bs),axis=-1,keepdims=True).astype('float32')
                combo = np.concatenate([bs, mask], axis=-1)
                combo = (combo - means.reshape(1,1,-1)) / (stds.reshape(1,1,-1) + 1e-6)
                Xb.append(np.nan_to_num(combo))
                coords.append((i,j))

        if Xb:
            probs = cnn.predict(np.stack(Xb), verbose=0).ravel()
            # fill any remaining holes in mc_map
            for (i,j), p in zip(coords, probs):
                mc_map[i,j] = 1.0 if p>=0.5 else 0.0

        # 6) early exit if no holes remain
        if (not np.isnan(rgb_norm[...,0]).any()) and (not np.isnan(mc_map).any()):
            break

    if rgb_norm is None:
        return None

    # mask out land
    rgb_norm = np.transpose(rgb_norm.copy(), axes = (1, 0, 2))
    rgb_norm[~lake_mask] = np.nan
    mc_map  [~lake_mask] = np.nan

    # ——— cache it for next time —————————————————————————————
    np.savez_compressed(
        cache_path,
        rgb_norm=rgb_norm,
        mc_map=mc_map,
        lat_c=lat_c,
        lon_c=lon_c,
        lake_mask=lake_mask
    )
    logging.info(f"Saved composite for {current_date} to cache")

    return rgb_norm, mc_map, lat_c, lon_c, lake_mask


# GUI setup
fig = plt.figure(figsize=(8,6))
ax_map = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
ax_radio = plt.axes([0.80, 0.55, 0.15, 0.30])
radio = RadioButtons(ax_radio, ('positive','negative'))
ax_prev = plt.axes([0.80, 0.40, 0.15, 0.05])
btn_prev = Button(ax_prev, 'Previous')
ax_next = plt.axes([0.80, 0.33, 0.15, 0.05])
btn_next = Button(ax_next, 'Confirm & Next')
rects = []

# Draw current date composite
def draw_map():
    global idx, _last_rgb, _last_mc_map, _last_lat_c, _last_lon_c, _last_lake_mask
    curr = dates_to_plot[idx]
    comp = compute_composite(curr)
    if comp is None:
        logging.info(f"Skipping {curr}: no data.")
        if idx < len(dates_to_plot)-1:
            idx += 1
            return draw_map()
        else:
            sys.exit(0)
    rgb_norm, mc_map, lat_c, lon_c, lake_mask = comp
    _last_rgb, _last_mc_map = rgb_norm, mc_map
    _last_lat_c, _last_lon_c, _last_lake_mask = lat_c, lon_c, lake_mask

    ax_map.clear()
    ax_map.set_extent([BBOX[0], BBOX[2], BBOX[1], BBOX[3]], ccrs.PlateCarree())
    # basemap
    ax_map.add_feature(cfeature.LAND.with_scale('10m'), facecolor='white', zorder=0)
    ax_map.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='black', zorder=0)
    # rgb on top
    ax_map.imshow(
        rgb_norm,
        origin='lower',
        extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]],
        transform=ccrs.PlateCarree(),
        interpolation='nearest',
        zorder=1
    )

    # Prob overlay
    lake_mask = _last_lake_mask
    mask = (~np.isnan(mc_map)) & (mc_map >= 0.5) & lake_mask

    rgba = np.zeros((*mc_map.shape, 4))
    rgba[..., 0][mask] = 1  # Red channel
    rgba[..., 3][mask] = 1  # Alpha channel

    ax_map.imshow(
        rgba,
        origin='lower',
        extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]],
        transform=ccrs.PlateCarree(),
        interpolation='nearest',
        zorder=4
    )
    
    ax_map.coastlines('10m')
    fig.suptitle(f"Date: {curr.isoformat()}")
    plt.draw()

# Click handler adds rectangle only
def on_click(event):
    if event.inaxes != ax_map:
        return
    lon, lat = event.xdata, event.ydata
    lbl = radio.value_selected
    lat_c, lon_c = _last_lat_c, _last_lon_c
    # find nearest cell
    i = np.argmin(np.abs(lat_c - lat))
    j = np.argmin(np.abs(lon_c - lon))
    # ensure over lake
    if not _last_lake_mask[i,j]:
        return
    # draw rectangle
    dl = abs(lon_c[1] - lon_c[0])
    dp = abs(lat_c[1] - lat_c[0])
    rect = Rectangle(
        (lon_c[j]-dl/2, lat_c[i]-dp/2),
        dl, dp,
        edgecolor='red' if lbl=='positive' else 'green',
        facecolor='none', linewidth=1.5,
        transform=ccrs.PlateCarree(), zorder=10
    )
    ax_map.add_patch(rect)
    rects.append(rect)
    all_labels.append({
        'date': dates_to_plot[idx].isoformat(),
        'lat': float(lat_c[i]),
        'lon': float(lon_c[j]),
        'label': lbl
    })
    plt.draw()

# Navigation
def on_prev(event):
    global idx, rects
    if idx > 0:
        idx -= 1
        for r in rects:
            r.remove()
        rects.clear()
        draw_map()

def on_next(event):
    global idx
    save_csv()
    if idx < len(dates_to_plot)-1:
        idx += 1
        draw_map()
    else:
        logging.info("Done. Exiting.")
        sys.exit(0)

# Connect events
fig.canvas.mpl_connect('button_press_event', on_click)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# Launch
if __name__ == '__main__':
    draw_map()
    plt.show()
