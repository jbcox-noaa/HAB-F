#!/usr/bin/env python3
"""
cnn_model_dashboard.py

1) Scans Grid_search_oversample7/3day_*px_*pm/model.keras for models,
   grouped by threshold (0.1, 5, 10).

2) For each date in a custom test range:
     - downloads PACE granules,
     - builds normalized RGB background (ny × nx × 3),
     - for each threshold, runs its models:
         extracts P×P patches in one vectorized batch,
         normalizes, predicts probabilities, accumulates into ny × nx maps,
     - caches a .npz per date via parallel processes.

3) Serves a Dash app to page through dates, pick threshold, and toggle RGB,
   with Prev/Next buttons, white land/black lake base, vertical flip,
   and city markers (labels for CLE & Sandusky below).
"""

import os, re, glob, logging
from datetime import date, datetime, timedelta, timezone
import numpy as np
import tensorflow as tf
import earthaccess
import xarray as xr
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.vectorized import contains as shp_contains
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import concurrent.futures

from helpers import (
    process_pace_granule,
    extract_datetime_from_filename,
    with_retries,
)

from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GRID_DIR       = "Grid_search_oversample7"
CACHE_DIR      = "cache"
DATA_DIR       = "data"
SAVE_DIR       = GRID_DIR
BBOX           = (-83.5, 41.3, -81.0, 42.75)
THRESHOLDS     = [0.1, 5, 10]
DEFAULT_THRESH = 0.1
MIN_VALID_FRAC = 0.5
DAYS_LOOKBACK  = 7
# ────────────────────────────────────────────────────────────────────────────────

def discover_models(grid_dir=GRID_DIR):
    models_by_thr = {thr: [] for thr in THRESHOLDS}
    pattern = os.path.join(grid_dir, "3day_*px_*pm", "model.keras")
    for path in glob.glob(pattern):
        m = re.search(r"3day_(\d+)px_([\d\.]+)pm", path)
        if not m:
            continue
        psize, thr = int(m.group(1)), float(m.group(2))
        if thr not in models_by_thr:
            continue
        mdl = tf.keras.models.load_model(path)
        mdl._patch_size = psize
        models_by_thr[thr].append(mdl)
    for thr in THRESHOLDS:
        if not models_by_thr[thr]:
            raise RuntimeError(f"No models found for threshold {thr}")
    return models_by_thr

def extract_patches_vectorized(arr, P, min_valid_frac):
    """
    arr: shape (bands, ny, nx)
    returns:
        coords: list of (i,j) center indices
        Xp:     array (n_patches, P, P, bands+1)
    """
    bands, ny, nx = arr.shape
    # sliding windows: (bands, ny-P+1, nx-P+1, P, P)
    win = sliding_window_view(arr, (1, P, P)).reshape(bands, ny-P+1, nx-P+1, P, P)
    win = np.moveaxis(win, 0, -1)  # → (ny-P+1, nx-P+1, P, P, bands)

    valid_cnt = np.count_nonzero(~np.isnan(win), axis=(2,3,4))
    total     = P*P*bands
    mask_good = (valid_cnt / total) >= min_valid_frac
    ys, xs    = np.nonzero(mask_good)
    if len(ys) == 0:
        return [], np.zeros((0, P, P, bands+1), dtype=float)

    patches = win[ys, xs]  # (n, P, P, bands)
    m4 = (~np.isnan(patches).any(axis=-1, keepdims=True)).astype("float32")
    Xp = np.concatenate([np.nan_to_num(patches, nan=0.0), m4], axis=-1)
    coords = [(y + P//2, x + P//2) for y, x in zip(ys, xs)]
    return coords, Xp

def compute_for_date(current_date, models_by_thr):
    # load normalization stats
    ch_means = np.load(os.path.join(DATA_DIR, "channel_means.npy"))
    ch_stds  = np.load(os.path.join(DATA_DIR, "channel_stds.npy"))
    ctx_means= np.load(os.path.join(SAVE_DIR, "context_means.npy"))
    ctx_stds = np.load(os.path.join(SAVE_DIR, "context_stds.npy"))

    # define window
    t1 = datetime.combine(current_date, datetime.min.time(), timezone.utc)
    t0 = t1 - timedelta(days=DAYS_LOOKBACK)
    # search & download
    r1 = with_retries(earthaccess.search_data, short_name="PACE_OCI_L2_AOP",
                      temporal=(t0,t1), bounding_box=BBOX)
    r2 = with_retries(earthaccess.search_data, short_name="PACE_OCI_L2_AOP_NRT",
                      temporal=(t0,t1), bounding_box=BBOX)
    results = r1 + r2
    if not results:
        raise RuntimeError(f"No granules for {current_date}")
    files = with_retries(earthaccess.download, results, "./data_cache/")
    files = sorted(files, key=lambda f: extract_datetime_from_filename(f), reverse=True)

    rgb_norm = None
    lake_mask = None
    lat_c = lon_c = None
    prob_sum   = {thr: None for thr in THRESHOLDS}
    prob_count = {thr: None for thr in THRESHOLDS}

    wl_ref = xr.open_dataset(
        os.path.join(DATA_DIR, "ref", "PACE_OCI.20240603T180158.L2.OC_AOP.V3_0.nc"),
        group="sensor_band_parameters"
    )["wavelength_3d"].data

    for f in files:
        try:
            wls, arr_stack, lat_arr, lon_arr = process_pace_granule(f, BBOX, {"res_km":1.2}, wl_ref)
        except:
            continue
        if arr_stack is None:
            continue

        # update RGB
        raw = np.dstack((arr_stack[105], arr_stack[75], arr_stack[48])).astype(float)
        mn, mx = np.nanmin(raw), np.nanmax(raw)
        if mx > mn:
            norm = (raw - mn) / (mx - mn)
            if rgb_norm is None:
                rgb_norm = norm.copy()
                lg, la = np.meshgrid(lon_arr, lat_arr)
                union = unary_union(list(cfeature.LAKES.with_scale("10m").geometries()))
                lake_mask = shp_contains(union, lg, la)
                lat_c, lon_c = lat_arr, lon_arr
            else:
                holes = np.isnan(rgb_norm[...,0])
                for c in range(3):
                    rgb_norm[...,c][holes] = norm[...,c][holes]

        gran_means = np.nanmean(arr_stack, axis=(1,2))
        ctx_norm   = (gran_means - ctx_means) / (ctx_stds + 1e-6)

        bands, ny, nx = arr_stack.shape
        for thr, models in models_by_thr.items():
            if prob_sum[thr] is None:
                prob_sum[thr]   = np.zeros((ny,nx), float)
                prob_count[thr] = np.zeros((ny,nx), float)

            for mdl in models:
                P = mdl._patch_size
                coords, Xp = extract_patches_vectorized(arr_stack, P, MIN_VALID_FRAC)
                if not coords:
                    continue
                # normalize spectral channels
                C = ch_means.shape[0]
                spec = (Xp[...,:C] - ch_means.reshape(1,1,1,C)) / (ch_stds.reshape(1,1,1,C)+1e-6)
                Xp[...,:C] = spec
                Xc = np.tile(ctx_norm, (len(Xp),1))
                probs = mdl.predict([Xp, Xc], verbose=0).ravel()
                for (i,j), p in zip(coords, probs):
                    prob_sum[thr][i,j]   += p
                    prob_count[thr][i,j] += 1

        if all((prob_count[thr]>0).all() for thr in THRESHOLDS):
            break

    if rgb_norm is None:
        raise RuntimeError(f"No RGB for {current_date}")

    # mask land
    land_mask = ~lake_mask
    rgb_plot = rgb_norm.copy()
    rgb_plot[land_mask,:] = np.nan

    prob_maps = {}
    for thr in THRESHOLDS:
        m = np.full_like(prob_sum[thr], np.nan)
        valid = prob_count[thr] > 0
        m[valid] = prob_sum[thr][valid] / prob_count[thr][valid]
        m[land_mask] = np.nan
        prob_maps[thr] = m

    return rgb_plot, prob_maps

def cache_one_date(dt):
    fn = os.path.join(CACHE_DIR, dt.strftime("%Y%m%d") + ".npz")
    if os.path.exists(fn):
        return dt, "skipped"
    models = discover_models()
    rgb, pmaps = compute_for_date(dt, models)
    np.savez_compressed(fn,
        rgb=rgb,
        prob01=pmaps[0.1],
        prob5=pmaps[5],
        prob10=pmaps[10]
    )
    return dt, "done"

def prepare_cache_parallel(dates, max_workers=4):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(cache_one_date, dt): dt for dt in dates}
        for fut in concurrent.futures.as_completed(futures):
            dt = futures[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                print(f"[{dt}] ERROR: {e}")
            else:
                print(f"[{dt}] → {status}")

def load_from_cache(dt):
    arr = np.load(os.path.join(CACHE_DIR, dt.strftime("%Y%m%d") + ".npz"))
    return arr["rgb"], {
        0.1: arr["prob01"],
        5:   arr["prob5"],
        10:  arr["prob10"]
    }

def create_dash_app(dates):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("Microcystin Ensemble Explorer"),
        html.Div([
            html.Button("← Prev", id="prev-btn"),
            html.Span(id="date-display", style={"margin":"0 1em"}),
            html.Button("Next →", id="next-btn"),
        ], style={"textAlign":"center","margin":"1em"}),
        html.Div([
            html.Label("Threshold (pm):"),
            dcc.RadioItems(
                id="thr-radio",
                options=[{"label":str(t),"value":t} for t in THRESHOLDS],
                value=DEFAULT_THRESH,
                inline=True
            ),
            html.Label(" Show RGB:"),
            dcc.Checklist(
                id="rgb-chk",
                options=[{"label":"RGB","value":"RGB"}],
                value=["RGB"]
            ),
        ], style={"textAlign":"center","margin":"1em"}),
        dcc.Graph(id="map-graph", config={"displayModeBar":False})
    ])

    @app.callback(
        Output("date-display","children"),
        Output("map-graph","figure"),
        Input("prev-btn","n_clicks"),
        Input("next-btn","n_clicks"),
        Input("thr-radio","value"),
        Input("rgb-chk","value"),
        prevent_initial_call=False
    )
    def update(prev, nxt, thr, rgb_opts):
        # ── figure index logic ─────────────────────────────────────────────
        ctx = callback_context
        if not ctx.triggered:
            idx = 0
        else:
            trig = ctx.triggered[0]["prop_id"].split(".")[0]
            idx = getattr(update, "idx", 0)
            if trig == "next-btn" and idx < len(dates)-1: idx += 1
            if trig == "prev-btn" and idx > 0:            idx -= 1
        update.idx = idx

        # ── load the cached arrays ──────────────────────────────────────────
        dt   = dates[idx]
        rgb, pmaps = load_from_cache(dt)
        probs = pmaps[thr]

        # ── composite main image ───────────────────────────────────────────
        alpha = probs[...,None]
        if "RGB" in rgb_opts:
            comp = rgb*(1-alpha) + alpha*np.array([1.0,0.0,0.0])
        else:
            comp = alpha*np.array([1.0,0.0,0.0])
        img = np.clip(comp*255, 0,255).astype(np.uint8)

        # ── build the white‐land black‐lake mask ────────────────────────────
        water = ~np.isnan(rgb).any(axis=-1)
        land  = ~water
        base  = np.zeros_like(img)
        base[land] = [255,255,255]
        final = base.copy()
        final[water] = img[water]

        # ── constants & margins ────────────────────────────────────────────
        lon_min, lat_min, lon_max, lat_max = BBOX
        extra = (lon_max - lon_min)*0.05  # width of the colorbar in lon‐units

        # ── compute background color for the bar ───────────────────────────
        if "RGB" in rgb_opts:
            # mean over all lake‐pixels to get an “avg RGB”
            mean_rgb = np.nanmean(rgb[water].reshape(-1,3),axis=0)
        else:
            mean_rgb = np.array([0,0,0],float)
        mean_rgb = (mean_rgb*255).astype(np.uint8)

        # ── build a gradient bar image (N × 1 × 3) ─────────────────────────
        N = final.shape[0]
        grads = np.linspace(1, 0, N)[:,None]   # from 1 at top to 0 at bottom
        bar = np.zeros((N, 1, 3), dtype=np.uint8)
        # each row = mean_bg*(1–g) + red*g
        for i,g in enumerate(grads[:,0]):
            bar[i,0,:] = (mean_rgb*(1-g) + np.array([255,0,0])*g).astype(np.uint8)

        # ── start the Figure and add everything in z‐order ────────────────
        fig = go.Figure()

        # 1) the color‐bar on the left
        fig.add_trace(go.Image(
            z=bar,
            x0=lon_min - extra,       # left edge
            y0=lat_min,               # bottom
            dx=extra,                 # width in lon units
            dy=(lat_max - lat_min)/N, # height per pixel
            colormodel="rgb"
        ))

        # 2) main RGB+overlay
        fig.add_trace(go.Image(
            z=final,
            x0=lon_min, y0=lat_min,
            dx=(lon_max-lon_min)/final.shape[1],
            dy=(lat_max-lat_min)/final.shape[0],
            colormodel="rgb"
        ))

        # 3) cities
        cities = {
            "Cleveland": (-81.6954,41.4993),
            "Sandusky":  (-82.7079,41.4489),
            "Toledo":    (-83.5552,41.6639),
            "Monroe":    (-83.3977,41.9164),
            "Colchester":(-82.8831,42.1756),
            "Leamington":(-82.5996,42.0500)
        }
        for name,(lon,lat) in cities.items():
            pos = "bottom center" if name in ("Cleveland","Sandusky") else "top center"
            fig.add_trace(go.Scatter(
                x=[lon], y=[lat],
                mode="markers+text",
                text=[name],
                textposition=pos,
                marker=dict(size=6, color="black"),
                showlegend=False
            ))

        # ── final layout tweaks ─────────────────────────────────────────────
        fig.update_layout(
            margin=dict(l=100,r=0,t=30,b=0),
            shapes=[  # if you still have lake‐shapes, they'd go here
                # …
            ],
            xaxis=dict(visible=False, showgrid=False,
                       range=[lon_min-extra, lon_max]),
            yaxis=dict(visible=False, showgrid=False, scaleanchor="x",
                       range=[lat_min, lat_max]),
            paper_bgcolor="white", plot_bgcolor="white",
            title=dict(text=dt.strftime("%Y-%m-%d"), x=0.5)
        )

        return dt.strftime("%Y-%m-%d"), fig


    return app

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    # custom test date

    # test dates only
    START_DATE_1 = date(2024, 5, 1)
    END_DATE_1   = date(2024, 11, 15)

    START_DATE_2 = date(2025, 6, 1)
    END_DATE_2   = date(2025, 8, 2)

    span_1 = (END_DATE_1 - START_DATE_1).days
    span_2 = (END_DATE_2 - START_DATE_2).days

    dates = [START_DATE_1+timedelta(days=i) for i in range(span_1 + 1)] + [START_DATE_2 + timedelta(days = i) for i in range(span_2 + 1)]

    prepare_cache_parallel(dates, max_workers=4)
    app = create_dash_app(dates)
    port = int(os.environ.get("PORT", 8080))

    app.run_server(host="0.0.0.0", port=port, debug=False)
