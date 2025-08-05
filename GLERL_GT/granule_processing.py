import os
import logging
import earthaccess

import pandas            as pd
import numpy             as np
import xarray            as xr
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import matplotlib.pyplot as plt

from helpers               import *
from matplotlib            import cm
from datetime              import datetime, timedelta
from cartopy.mpl.ticker    import LongitudeFormatter, LatitudeFormatter


CORRUPTED_GRANULES = set([
    "PACE_OCI.20240415T172837.L2.OC_AOP.V3_0.nc",
    "PACE_OCI.20240504T170438.L2.OC_AOP.V3_0.nc"
])
CORRUPTED_FILE_PATH = "corrupted_granules.txt"

def load_corrupted_granules(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r") as f:
        return set(line.strip() for line in f if line.strip())

CORRUPTED_GRANULES.update(load_corrupted_granules(CORRUPTED_FILE_PATH))

def process_granule(
    filepath: str,
    station_df: pd.DataFrame,
    bbox: tuple,
    sensor_params: dict,
    results: list,
    station_colors: dict,
    results_path: str,
    images_root: str,
    wave_all: np.ndarray,
    half_time_window: int,
    patch_size: int,
    pm_threshold: float
) -> None:
    """
    Extract and record a single granule’s patches for a given patch_size,
    recording valid_frac and patch_size without skipping by coverage.
    """
    filename = os.path.basename(filepath)

    dt = extract_datetime_from_filename(filename)
    if dt is None:
        logging.warning(f"Bad filename {filename}")
        return

    # Only PACE flow shown; extend for other sensors as needed
    try:
        wls, arr_stack, target_lats, target_lons = process_pace_granule(
            filepath, bbox, sensor_params, wave_all
        ) or (None, None, None, None)
    except RuntimeError as e:
        print(f"Error during extraction: {e}")
        print(f"Marking granule as corrupted: {filename}")
        CORRUPTED_GRANULES.add(filename)
        with open(CORRUPTED_FILE_PATH, "a") as f:
            f.write(filename + "\n")
        return  # skip processing this granule
    
    if arr_stack is None:
        return

    global_means = np.nanmean(arr_stack, axis=(1,2))
    t0 = pd.to_datetime(dt).tz_localize('UTC')
    window_start = t0 - timedelta(days=half_time_window)
    window_end = t0 + timedelta(days=half_time_window)
    df_window = station_df[
        (station_df.timestamp >= window_start) &
        (station_df.timestamp <= window_end)
    ]
    if df_window.empty:
        return
    
    res_km = sensor_params["res_km"]
    res_lat_deg = res_km / 111.0

    # 2) lon spacing needs cosine correction at mid‐latitude
    lon_min, lat_min, lon_max, lat_max = bbox
    lat0_center = (lat_min + lat_max) / 2.0
    res_lon_deg = res_km / (111.0 * math.cos(math.radians(lat0_center)))

    # 3) build bin edges then centers
    lat_bins = np.arange(lat_min, lat_max + res_lat_deg, res_lat_deg)
    lon_bins = np.arange(lon_min, lon_max + res_lon_deg, res_lon_deg)

    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

    # Loop stations
    for station in df_window.station_name.unique():
        df_st = df_window[df_window.station_name == station]
        lat0, lon0 = estimate_position(
            df_st.timestamp.tolist(), df_st.lat.values, df_st.lon.values, t0
        )
        patch_dict = extract_pace_patch(
            arr_stack, wls, lon0, lat0,
            pixel_count=patch_size,
            lat_centers=lat_centers, lon_centers=lon_centers
        )
        # compute coverage fraction
        total = sum(arr.size for arr in patch_dict.values())
        valid = sum(np.count_nonzero(~np.isnan(arr)) for arr in patch_dict.values())
        valid_frac = valid/total if total else 0.0

        # Build feature & label
        stack = np.stack([patch_dict[wl] for wl in sorted(patch_dict)], axis=-1)
        fv = np.concatenate([stack.flatten(), global_means])
        label_tuple = (
            station, t0, lat0, lon0,
            df_st.particulate_microcystin.mean(),
            df_st.dissolved_microcystin.mean() if 'dissolved_microcystin' in df_st else np.nan,
            df_st.extracted_chla.mean() if 'extracted_chla' in df_st else np.nan
        )
        # Append full record including valid_frac & patch_size
        results.append((filename, station, label_tuple, fv, valid_frac, patch_size))

    # save
    np.save(results_path, np.array(results, dtype=object))
    save_station_colors(sensor_params['station_colors_json'], station_colors)

def process_all_granules(
        sensor: str = "PACE",
        patch_sizes: list[int] = [3, 5, 7, 9],
        half_time_window: int = 2,
        pm_threshold: float = 0.1,
        save_dir: str = './',
        load_user_df: bool = False
    ) -> None:
    """
    Download and extract patches for multiple patch_sizes in one pass.

    This collects ALL patches (no coverage filtering) and saves a single
    numpy array of tuples:
      (filename, station, label_tuple, feature_vector, valid_frac, patch_size)
    to:
      {save_dir}/training_data_{sensor}.npy
    """
    configure_logging()
    logging.info(f"Starting multi-size extraction for sensor={sensor}")

    # --- Load station data
    df = pd.read_csv("glrl-hab-data.csv", index_col=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if load_user_df:
        user_df = pd.read_csv("user-labels.csv", index_col=False)
        user_df["timestamp"] = pd.to_datetime(user_df["date"], utc=True)
        user_df["station_name"] = [f"USER_{i}" for i in range(len(user_df))]
        user_df.loc[user_df["label"]=='negative', 'particulate_microcystin'] = 0
        user_df.loc[user_df["label"]=='positive', 'particulate_microcystin'] = pm_threshold
        df = pd.concat([df, user_df], ignore_index=True)

    # Determine download window
    latest_ts = df["timestamp"].max()
    if pd.isna(latest_ts):
        logging.warning("No valid timestamps; abort extraction.")
        return
    end_date = latest_ts.date()

    # sensor mapping
    sensor = sensor.upper()
    sensor_map = {
        "PACE": {"short_names": ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"],
                 "start_date": datetime(2024,4,16), "res_km": 1.2},
        # ... other sensors omitted for brevity ...
    }
    if sensor not in sensor_map:
        raise ValueError(f"Unsupported sensor '{sensor}'")
    params = sensor_map[sensor]
    params["sensor"] = sensor
    params["bbox"] = (-83.5, 41.3, -82.45, 42.2)

    # Prepare output paths
    extraction_path = os.path.join(save_dir, f"training_data_{sensor}.npy")
    station_colors_json = os.path.join(save_dir, f"station_colors_{sensor}.json")
    params["station_colors_json"] = station_colors_json

    # Initialize or load
    if os.path.exists(extraction_path):
        existing = np.load(extraction_path, allow_pickle=True)
        results = existing.tolist()
    else:
        results = []
    station_colors = load_station_colors(station_colors_json)

    # Auth and time loop
    auth = earthaccess.login(persist=True)
    start_date = params["start_date"].date() if isinstance(params["start_date"], datetime) else params["start_date"]
    if start_date > end_date:
        logging.info("No data window to process.")
        return

    # Loop by month
    month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")
    data_root = os.path.join("./data", sensor)
    os.makedirs(data_root, exist_ok=True)

    # right after you’ve logged in and created your data directory:
    print("Retrieving wavelength list from a reference file…")
    ref_search = with_retries(
        earthaccess.search_data,
        short_name="PACE_OCI_L2_AOP",
        temporal=("2024-06-01", "2024-06-05"),
        bounding_box=params["bbox"],
    )
    ref_file = with_retries(earthaccess.download, ref_search, "./data")[0]
    wave_all = xr.open_dataset(ref_file, group="sensor_band_parameters")["wavelength_3d"].data


    for ms in month_starts.to_pydatetime():
        year, month = ms.year, ms.month
        month_start = datetime(year, month, 1)
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        if next_month.date() > end_date:
            next_month = datetime(end_date.year, end_date.month, end_date.day) + timedelta(days=1)

        start_iso = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso   = next_month.strftime("%Y-%m-%dT%H:%M:%SZ")
        logging.info(f"Searching {sensor} granules {year}-{month:02d}")
        all_search = []
        for sn in params["short_names"]:
            try:
            
                res = with_retries(
                    earthaccess.search_data,
                    short_name=sn,
                    temporal=(start_iso, end_iso),
                    bounding_box=params["bbox"]
                )
            
                if res:
                    logging.info(f"Found {len(res)} granules for short_name={sn} in {year}-{month:02d}")
                    all_search.extend(res)
                else:
                    logging.info(f"No granules for short_name={sn} in {year}-{month:02d}")
            
            except Exception as e:
                logging.error(f"Search failed for {sn} in {year}-{month:02d}: {e}")
        
        if not all_search:
            logging.info(f"No granules found for month {year}-{month:02d}.")
            continue
        
        filtered_search = []
        skipped_processed = 0
        skipped_no_obs = 0

        processed_set = set()

        for item in all_search:
            fname = get_granule_filename(item)
            if fname is None:
                logging.warning("Cannot determine filename for item; skipping")
                continue

            # 2. Determine granule datetime for station-window filtering
            dt = None
            if hasattr(item, "time_start"):
                try:
                    dt = pd.to_datetime(item.time_start, utc=True)
                except:
                    dt = None
            if dt is None:
                # fallback: parse from fname
                dt0 = extract_datetime_from_filename(fname)
                if dt0 is not None:
                    dt = pd.to_datetime(dt0).tz_localize("UTC")
            if dt is None:
                logging.info(f"Skipping granule {fname}: cannot get datetime")
                continue

            # 3. Check for any station timestamps within ±2 days
            window_start = dt - pd.Timedelta(days=2)
            window_end   = dt + pd.Timedelta(days=2)
            if not ((df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)).any():
                skipped_no_obs += 1
                continue

            # If passed both checks, keep this item
            filtered_search.append(item)

        logging.info(f"Skipped {skipped_processed} already-processed granules; "
                    f"{skipped_no_obs} with no nearby observations; "
                    f"{len(filtered_search)} remain for download.")

        if not filtered_search:
            logging.info("No relevant new granules to download for this month.")
            continue

        # Now download only filtered_search
        try:
            print()
            print("Downloading filtered search")
            paths = with_retries(
                earthaccess.download,
                filtered_search,
                './data'
            )
            
        except Exception as e:
            logging.error(f"Download failed: {e}")
            continue

        CORRUPTED_GRANULES = {
            "data/PACE_OCI.20240506T181430.L2.OC_AOP.V3_0.nc",
            "data/PACE_OCI.20240506T180930.L2.OC_AOP.V3_0.nc",
            "data/PACE_OCI.20240706T190615.L2.OC_AOP.V3_0.nc"
        }

        # Process each granule for each patch_size
        for filepath in paths:

            if filepath in CORRUPTED_GRANULES:
                print(f"Skipping corrupted granule {filepath}")
                continue

            for ps in patch_sizes:
                process_granule(
                    filepath=filepath,
                    station_df=df,
                    bbox=params.get("bbox", (-83.5,41.3,-82.45,42.2)),
                    sensor_params=params,
                    results=results,
                    station_colors=station_colors,
                    results_path=extraction_path,
                    images_root=os.path.join(save_dir, "Images/Patches"),
                    wave_all=wave_all,  # your ref loading logic omitted
                    half_time_window=half_time_window,
                    patch_size=ps,
                    pm_threshold=pm_threshold
                )

    # Final save
    np.save(extraction_path, np.array(results, dtype=object))
    save_station_colors(station_colors_json, station_colors)
    logging.info(f"Extraction complete: {len(results)} patches saved to {extraction_path}")
