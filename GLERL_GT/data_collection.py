import os
import logging

from datetime              import date
from granule_processing    import process_all_granules
from balance_training_data import balance_dataset_by_granule
from oversample            import balance_dataset_by_oversample
from plot_modeling         import plot_model_results
from train_model           import train_model
    
logging.basicConfig(level=logging.INFO)

bbox = (-83.5, 41.3, -82.45, 42.2)

patch_sizes       = [5, 3, 7, 9]
half_time_windows = [1]
pm_thresholds     = [0.1]

dates_to_plot = [
    date(2024, 5, 9),
    date(2024, 7, 8),
    date(2024, 7, 12),
    date(2024, 7, 13),
    date(2024, 7, 14),
    date(2024, 7, 19),
    date(2024, 7, 20),
    date(2024, 8, 2),
    date(2024, 8, 14),
    date(2024, 9, 3),
    date(2024, 9, 16)
]

for patch_size in patch_sizes:
    for half_time_window in half_time_windows:
        for pm_threshold in pm_thresholds:

            save_dir = f'Grid_search_oversample5/{(half_time_window * 2) + 1}day_{patch_size}px_{pm_threshold}pm/'
            os.makedirs(save_dir, exist_ok=True)

            process_all_granules(
                half_time_window = half_time_window, # days
                patch_size       = patch_size,       # pixels 
                pm_threshold     = pm_threshold,
                save_dir         = save_dir,
                load_user_df     = True
            ) 

            """ balance_dataset_by_granule(
                patch_size       = patch_size,       # pixels
                pm_threshold     = pm_threshold,     # micrograms / liter
                save_dir         = save_dir
            ) """

            balance_dataset_by_oversample(
                patch_size       = patch_size,
                pm_threshold     = pm_threshold,
                bbox             = bbox,
                neg_to_pos_ratio = 20,
                start_date       = date(2024, 11, 15),
                end_date         = date(2025, 5, 1),
                save_dir         = save_dir
            )
            input()

            train_model(
                save_dir         = save_dir,
                sensor           = "PACE",
                patch_size       = patch_size,
                pm_threshold     = pm_threshold
            )

            plot_model_results(
                patch_size       = patch_size,
                save_dir         = save_dir,
                dates_to_plot    = None
            )