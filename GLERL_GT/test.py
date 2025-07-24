#!/usr/bin/env python3
"""
cnn_labeling_labeler.py
Standalone Matplotlib+Cartopy GUI for interactive labeling of misclassified pixels over lakes.
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

# Helpers (ensure these are defined in helpers.py)
from helpers import (
    process_pace_granule,
    extract_datetime_from_filename,
    extract_pace_patch,
    with_retries
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
START_DATE     = date(2024, 4, 15)
END_DATE       = date(2024, 10, 15)
LOOKBACK_DAYS  = 1
PATCH_SIZE     = 5
# (west, south, east, north)
BBOX           = (-83.5, 41.3, -82.45, 42.2)
DATA_DIR       = './data/'
CSV_PATH       = './user-labels.csv'
MODEL_PATH     = './models/model.keras'
MEANS_PATH     = os.path.join('data', 'channel_means.npy')
STDS_PATH      = os.path.join('data', 'channel_stds.npy')
WAVELENGTH_REF = os.path.join('data', 'ref', 'PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc')

# Pre-load model and stats
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

# Dates and label store
dates_to_plot = [START_DATE + timedelta(days=i)
                 for i in range((END_DATE - START_DATE).days + 1)]
all_labels = []
idx = 0

# Globals cache last composite + grid + lake mask
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
    logging.info(f"Saved {len(all_labels)} labels → {CSV_PATH}")

# Compute composite and pre-mask over lake
def compute_composite(current_date):
    t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)
    t0 = t1 - timedelta(days=LOOKBACK_DAYS)
    logging.info(f"Searching {t0.isoformat()} to {t1.isoformat()}")
    # search data
    r1 = with_retries(earthaccess.search_data,
                       short_name='PACE_OCI_L2_AOP', temporal=(t0,t1), bounding_box=BBOX)
    r2 = with_retries(earthaccess.search_data,
                       short_name='PACE_OCI_L2_AOP_NRT', temporal=(t0,t1), bounding_box=BBOX)
    results = r1 + r2
    if not results:
        return None
    # download
    files = with_retries(earthaccess.download, results, DATA_DIR)
    files_sorted = sorted(files, key=extract_datetime_from_filename, reverse=True)

    # hole-fill combine
    pacearray = None
    for fpath in files_sorted:
        try:
            wls, arr, lat_c, lon_c = process_pace_granule(fpath, BBOX, {'res_km':1.2}, wave_all)
        except Exception:
            continue
        if pacearray is None:
            pacearray = arr.copy()
        else:
            m = ~np.isnan(arr)
            pacearray[m & np.isnan(pacearray)] = arr[m & np.isnan(pacearray)]
        if not np.isnan(pacearray).any(): break
    if pacearray is None:
        return None

    # build rgb_norm and flip axes
    rgb = np.dstack([pacearray[i] for i in (105,75,48)])
    rgb = np.transpose(rgb, (1,0,2))
    mn, mx = np.nanmin(rgb), np.nanmax(rgb)
    rgb_norm = (rgb - mn) / (mx - mn)

    # cnn prob map
    patches, coords = [], []
    for i, la in enumerate(lat_c):
        for j, lo in enumerate(lon_c):
            pd = extract_pace_patch(pacearray, wls, lo, la, PATCH_SIZE, lat_c, lon_c)
            bs = np.stack([pd[wl] for wl in wls], axis=-1)
            mask_arr = np.any(~np.isnan(bs), axis=-1, keepdims=True).astype('float32')
            if not mask_arr.any(): continue
            combo = np.concatenate([bs, mask_arr], axis=-1)
            combo = (combo - means.reshape(1,1,-1)) / (stds.reshape(1,1,-1)+1e-6)
            patches.append(np.nan_to_num(combo))
            coords.append((i,j))
    mc_map = np.full((len(lat_c), len(lon_c)), np.nan)
    if patches:
        X = np.stack(patches)
        probs = cnn.predict(X).ravel()
        for (i,j), p in zip(coords, probs): mc_map[i,j] = p

    # build lake mask: True inside lake polygons
    lake_polys = list(cfeature.LAKES.geometries())
    # grid in lat/lon order lat_c (rows), lon_c (cols)
    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
    mask = np.zeros_like(lon_grid, dtype=bool)
    for poly in lake_polys:
        for ii in range(mask.shape[0]):
            for jj in range(mask.shape[1]):
                if poly.contains(Point(lon_grid[ii,jj], lat_grid[ii,jj])):
                    mask[ii,jj] = True

    # apply mask
    rgb_norm[~mask] = np.nan
    mc_map[~mask] = np.nan

    return rgb_norm, mc_map, lat_c, lon_c, mask

# GUI setup
fig = plt.figure(figsize=(8,6))
ax_map = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
ax_radio = plt.axes([0.80,0.55,0.15,0.30])
radio = RadioButtons(ax_radio, ('positive','negative'))
ax_prev = plt.axes([0.80,0.40,0.15,0.05])
btn_prev = Button(ax_prev, 'Previous')
ax_next = plt.axes([0.80,0.33,0.15,0.05])
btn_next = Button(ax_next, 'Confirm & Next')
rects = []

# draw map handler
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
    ax_map.set_extent(BBOX, ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND.with_scale('10m'), facecolor='white', zorder=0)
    ax_map.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='black', zorder=0)

    # rgb and prob overlays above basemap
    ax_map.imshow(rgb_norm, origin='lower', extent=[BBOX[0],BBOX[2],BBOX[1],BBOX[3]],
                  transform=ccrs.PlateCarree(), interpolation='nearest', zorder=1)
    mask_o = (~np.isnan(mc_map))
    rgba = np.zeros((*mc_map.shape,4))
    rgba[...,0][mask_o] = 1; rgba[...,3][mask_o] = 1
    ax_map.imshow(
        rgba,
        origin = 'lower',
        extent = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]],
        transform = ccrs.PlateCarree(),
        interpolation = 'nearest',
        zorder = 2
    )
    
    ax_map.coastlines('10m')
    fig.suptitle(f"Date: {curr.isoformat()}")
    plt.draw()

# click handler
def on_click(event):
    if event.inaxes != ax_map: return
    lon, lat = event.xdata, event.ydata
    lbl = radio.value_selected
    lat_c, lon_c = _last_lat_c, _last_lon_c
    # find nearest cell
    i = np.argmin(np.abs(lat_c - lat))
    j = np.argmin(np.abs(lon_c - lon))
    # ensure over lake
    if not _last_lake_mask[j, i]: return
    # draw rectangle
    dl = abs(lon_c[1]-lon_c[0]); dp = abs(lat_c[1]-lat_c[0])
    rect = Rectangle((lon_c[j]-dl/2, lat_c[i]-dp/2), dl, dp,
                     edgecolor='red' if lbl=='positive' else 'green',
                     facecolor='none', linewidth=1.5,
                     transform=ccrs.PlateCarree(), zorder=3)
    ax_map.add_patch(rect)
    rects.append(rect)
    all_labels.append({'date': dates_to_plot[idx].isoformat(),
                       'lat': float(lat_c[i]), 'lon': float(lon_c[j]),
                       'label': lbl})
    plt.draw()

# navigation handlers
def on_prev(event):
    global idx
    if idx>0:
        idx -=1
        for r in rects: r.remove()
        rects.clear()
        draw_map()

def on_next(event):
    global idx
    save_csv()
    if idx < len(dates_to_plot)-1:
        idx +=1
        draw_map()
    else:
        logging.info("Completed all dates.")
        sys.exit(0)

# connect
fig.canvas.mpl_connect('button_press_event', on_click)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# start
if __name__ == '__main__':
    draw_map()
    plt.show()
