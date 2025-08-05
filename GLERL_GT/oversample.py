import os
import glob
import math
import logging
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess
import cartopy.feature as cfeature
from shapely.ops import unary_union
from shapely.geometry import Point
from earthaccess import Auth, DataCollections, DataGranules, Store, download
from pathlib import Path
import earthaccess


from helpers import (
    process_pace_granule,
    extract_pace_patch,
    get_granule_filename,
    with_retries
)

def shp_contains(lake_geom, lon_grid, lat_grid):
    """
    Return boolean mask matching lon_grid/lat_grid where points are inside lake_geom.
    """
    contains_vec = np.vectorize(lambda lon, lat: lake_geom.contains(Point(lon, lat)))
    return contains_vec(lon_grid, lat_grid)


def balance_dataset_by_oversample(
        sensor            = "PACE",
        patch_size        = 3,
        pm_threshold      = 0.1,
        bbox              = (-83.5, 41.3, -82.45, 42.2),
        neg_samples       = 3,
        start_date        = None,
        end_date          = None,
        save_dir          = './'
    ):
    """
    Oversample negative patches from across the lake to achieve a custom negative samples number,
    appending per-band global means to each patch vector.
    """
    logging.basicConfig(level=logging.INFO)
    rng = random.Random(45)

    # --- Load existing samples (positives & any negatives) ---
    pos_path = os.path.join(save_dir, f'training_data_{sensor}.npy')
    if not os.path.exists(pos_path):
        logging.error(f"No existing training data at {pos_path}; run process_all_granules first.")
        return
    data = np.load(pos_path, allow_pickle=True).tolist()

    # split by label
    positives = [r for r in data if r[2][4] >= pm_threshold]
    negatives = [r for r in data if r[2][4] <  pm_threshold]
    num_pos     = len(positives)
    existing_neg = len(negatives)
    if num_pos == 0:
        logging.info("No positive samples found; nothing to balance.")
        return

    # determine how many new negatives are needed
    total_target_neg = neg_samples
    need_neg = max(0, total_target_neg - existing_neg)
    logging.info(f"Found {num_pos} positives, {existing_neg} negatives, need {need_neg} more.")

    if need_neg == 0:
        logging.info("Already have sufficient negatives; no oversampling required.")
        return

    # --- Load reference wavelengths & build lake mask ---
    logging.info("Loading reference wavelengths...")
        # right after you’ve logged in and created your data directory:
    print("Retrieving wavelength list from a reference file…")
    ref_search = with_retries(
        earthaccess.search_data,
        short_name="PACE_OCI_L2_AOP",
        temporal=("2024-06-01", "2024-06-05"),
        bounding_box=bbox,
    )
    ref_file = with_retries(earthaccess.download, ref_search, "./data")[0]
    wave_all = xr.open_dataset(ref_file, group="sensor_band_parameters")["wavelength_3d"].data
    n_wl = wave_all.size

    _, arr_stack_ref, lat_grid_ref, lon_grid_ref = process_pace_granule(
        ref_file, bbox,  {"res_km":1.2}, wave_all
    ) or (None, None, None, None)

    if arr_stack_ref is None:
        # os.remove(filepath)
        print("Ref file is bad.")
        return

    lon_grid, lat_grid = np.meshgrid(lon_grid_ref, lat_grid_ref)
    raw_geoms = list(cfeature.LAKES.with_scale('10m').geometries())
    lake_union = unary_union([g for g in raw_geoms if hasattr(g, 'geom_type')])
    lake_mask = shp_contains(lake_union, lon_grid, lat_grid)

    # --- Find candidate granules ---
    granule_items = []
    def find_in_range(start_iso, end_iso):
        for sn in ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"]:
            try:
                res = earthaccess.search_data(
                    short_name=sn,
                    temporal=(start_iso, end_iso),
                    bounding_box=bbox
                )
                if res:
                    granule_items.extend(res)
            except Exception as e:
                logging.warning(f"Search failed for {sn}: {e}")

    if start_date and end_date:
        s = start_date.strftime if not isinstance(start_date, str) else lambda: start_date
        e = end_date.strftime   if not isinstance(end_date,   str) else lambda: end_date
        find_in_range(s("%Y-%m-%dT00:00:00Z"), e("%Y-%m-%dT00:00:00Z"))
    else:
        years = set(pd.to_datetime([r[2][1] for r in positives], utc=True).year)
        for yr in years:
            for m in [12,1,2,3]:
                st = datetime(yr, m, 1)
                en = (st.replace(month=m%12+1, year=yr + (m==12)) 
                      if m<12 else datetime(yr+1,1,1))
                find_in_range(st.strftime("%Y-%m-%dT00:00:00Z"),
                              en.strftime("%Y-%m-%dT00:00:00Z"))

    if not granule_items:
        logging.error("No granules found in specified period.")
        return

    # uniquify & shuffle
    unique = {get_granule_filename(it): it for it in granule_items}
    items = list(unique.items())
    rng.shuffle(items)

    # load invalid set
    invalid_txt = os.path.join(save_dir, f'invalid_granules_{sensor}.txt')
    invalid_set = set(open(invalid_txt).read().split()) if os.path.exists(invalid_txt) else set()

    # --- Oversample negatives ---
    balanced = list(data)
    added, idx, attempts = 0, 0, 0
    max_attempts = len(items) * 100

    while added < need_neg and attempts < max_attempts:
        fname, item = items[idx]
        idx = (idx + 1) % len(items)
        attempts += 1

        if fname in invalid_set:
            continue

        try:
            paths = earthaccess.download([item], './data')
            gran   = paths[0]
            res = process_pace_granule(gran, bbox, {"res_km":1.2}, wave_all)
        except Exception as e:
            logging.warning(f"Failed {fname}: {e}")
            with open(invalid_txt,'a') as f:
                f.write(fname+"\n")
            invalid_set.add(fname)
            continue

        if res is None:
            invalid_set.add(fname)
            continue

        # unpack
        _, arr_stack, lat_grid, lon_grid = res

        # pick a random lake cell
        flat_idx = rng.choice(np.flatnonzero(lake_mask))
        i, j     = np.unravel_index(flat_idx, lake_mask.shape)
        lon0, lat0 = lon_grid[j], lat_grid[i]

        patch = extract_pace_patch(
            arr_stack, wave_all,
            lon0, lat0,
            patch_size,
            lat_centers=lat_grid, lon_centers=lon_grid
        )
        if not patch or all(np.isnan(a).all() for a in patch.values()):
            continue

        # valid‐pixel check
        total_pixels = sum(a.size for a in patch.values())
        valid_pixels = sum(np.count_nonzero(~np.isnan(a)) for a in patch.values())
        if valid_pixels / total_pixels < 0.4:
            continue

        # build the patch vector
        bands = sorted(patch.keys())
        stack = np.stack([patch[b] for b in bands], axis=-1)  # (px,px,n_wl)
        flat  = stack.flatten()                               # length = px*px * n_wl

        # compute & append granule‐wide means
        global_means   = np.nanmean(arr_stack, axis=(1,2))    # (n_wl,)
        flat_with_ctx = np.concatenate([flat, global_means])  # flat + n_wl

        # label = ('RAND', timestamp, lon/lat... zero microcystin)
        t0    = datetime.utcnow()
        label = ('RAND', t0, lat0, lon0, 0.0, np.nan)

        balanced.append((fname, 'RAND', label, flat_with_ctx))
        added += 1
        logging.info(f"Added {added}/{need_neg}: {fname} @ {lon0:.3f},{lat0:.3f}")

    # --- Save balanced dataset ---
    out_path = os.path.join(save_dir, f'training_data_balanced_{sensor}.npy')
    np.save(out_path, np.array(balanced, dtype=object))
    logging.info(f"Done: added {added} negatives → saved {out_path}")
