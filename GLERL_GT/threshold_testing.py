import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# Import the multi-size extractor, oversampler, and trainer
from granule_processing import process_all_granules
from oversample import balance_dataset_by_oversample
from train_model import train_model

# === CONFIGURATION ===
SENSOR = "PACE"
PATCH_SIZES = [3, 5, 7, 9]
HALF_TIME_WINDOW = 2            # days for extraction
PM_THRESHOLD = 0.1              # label threshold
NEG_SAMPLES = 1000               # for oversampling
OV_START = date(2024, 11, 15)
OV_END   = date(2025, 5, 1)

SAVE_DIR = './all_patches/'     # base directory
VALID_FRAC_THRESHOLDS = np.arange(0.1, 1.0, 0.1)
BBOX = (-83.5, 41.3, -81, 42.75)
RES_LON_DEG = 0.005             # horizontal spatial resolution (used elsewhere)

# === STEP 1: EXTRACTION ===
os.makedirs(SAVE_DIR, exist_ok=True)
print("=== Step 1: Extracting all patches ===")

try:
    EXTRACTION_FILE = os.path.join(SAVE_DIR, f"training_data_{SENSOR}.npy")
    if not os.path.exists(EXTRACTION_FILE):
        process_all_granules(
            sensor=SENSOR,
            patch_sizes=PATCH_SIZES,
            half_time_window=HALF_TIME_WINDOW,
            pm_threshold=PM_THRESHOLD,
            save_dir=SAVE_DIR,
            load_user_df=True
        )
        print(f"Extraction complete: {EXTRACTION_FILE}")
    else:
        print(f"Extraction file already exists: {EXTRACTION_FILE}, skipping extraction.")
except Exception as e:
    print(f"Error during extraction: {e}")
    raise

# === STEP 2: GRID SEARCH WITH OVERSAMPLING ===
all_data = np.load(EXTRACTION_FILE, allow_pickle=True)
records = []

for ps in PATCH_SIZES:
    subset = [e for e in all_data if e[5] == ps]
    for thr in VALID_FRAC_THRESHOLDS:
        filtered = [e for e in subset if e[4] >= thr]
        if not filtered:
            continue
        # prepare directory for this patch_size / threshold
        save_dir_thr = os.path.join(SAVE_DIR, 'grid_results', f'{ps}px', f'{int(thr*100):02d}')
        os.makedirs(save_dir_thr, exist_ok=True)
        # save raw filtered extraction
        np.save(
            os.path.join(save_dir_thr, f'training_data_{SENSOR}.npy'),
            np.array(filtered, dtype=object)
        )
        # === Oversample to balance dataset ===
        print(f"Oversampling for patch={ps}, thr={thr}")
        balance_dataset_by_oversample(
            patch_size=ps,
            pm_threshold=PM_THRESHOLD,
            bbox=BBOX,
            neg_samples=NEG_SAMPLES,
            start_date=OV_START,
            end_date=OV_END,
            save_dir=save_dir_thr
        )
        # Now train on the oversampled balanced data
        print(f"Training model for patch={ps}, thr={thr}")
        loss, acc, auc, f1 = train_model(
            save_dir=save_dir_thr,
            sensor=SENSOR,
            patch_size=ps,
            pm_threshold=PM_THRESHOLD
        )
        records.append({
            'patch_size': ps,
            'valid_frac': thr,
            'f1_score': f1
        })

# === STEP 3: SAVE & PLOT RESULTS ===
df = pd.DataFrame(records)
out_csv = os.path.join(SAVE_DIR, 'grid_search_results.csv')
out_png = os.path.join(SAVE_DIR, 'grid_search_plot.png')
df.to_csv(out_csv, index=False)

plt.figure()
for ps in PATCH_SIZES:
    sub = df[df.patch_size == ps]
    plt.plot(sub.valid_frac, sub.f1_score, marker='o', label=f'{ps}px')
plt.xlabel('Valid Fraction Threshold')
plt.ylabel('Test F1 Score')
plt.title('F1 vs Coverage Threshold by Patch Size')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_png)
plt.show()

print(f"Grid search complete. Results: {out_csv}, Plot: {out_png}")
