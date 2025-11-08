"""explorer.py

Module dedicated to the visual exploration of the raw dataset's characteristics.

This module acts as a dispatcher, selecting the appropriate file paths based on
the data generation mode (static, dynamic, or real) and invoking specialized
plotting functions to analyze and visualize the dataset's properties. These
functions are invoked remotely leveraging the Ray framework.

Functions:
    explore_data(
        timestamps_hours: np.ndarray,
        requests: list[int],
        min_key: int,
        max_key: int,
        data_mode: str,
    ) -> tuple[str, str, str]
        Generates and saves a set of plots for the given dataset.
"""

import numpy as np

from components.logs.levels.debug_logger import debug
from components.ray.tasks.data.explorer import (
    plot_zipf_loglog_task,
    plot_daily_profile_task,
    plot_key_usage_heatmap_task,
)
from const import (
    DATA_STATIC_MODE,
    DATA_DYNAMIC_MODE,
    DATA_REAL_MODE,
)
from pipeline.const import (
    PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH,
    PLOT_STATIC_DAILY_PROFILE_FILE_PATH,
    PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH,
    PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH,
    PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_REAL_ZIPF_LOG_LOG_FILE_PATH,
    PLOT_REAL_DAILY_PROFILE_FILE_PATH,
    PLOT_REAL_KEY_USAGE_HEATMAP_FILE_PATH,
)


def explore_data(
    timestamps_hours: np.ndarray,
    requests: list[int],
    min_key: int,
    max_key: int,
    data_mode: str,
) -> tuple[str, str, str]:
    """Generates and saves a set of plots for the given dataset.

    This function calls dedicated plotting utilities to create and
    save three plots: the Zipf distribution, the daily request profile,
    and a key usage heatmap over time.

    Args:
        timestamps_hours (np.ndarray): An array of request timestamps.
        requests (list[int]): A list of requested keys.
        min_key (int): The minimum key in the dataset.
        max_key (int): The maximum key in the dataset.
        data_mode (str): The mode of the data.

    Returns:
        tuple[str, str, str]:
            - zipf_log_log_plot_save_path: Path Zipf log-log plot is saved to.
            - daily_profile_plot_save_path: Path daily profile plot is saved to.
            - key_usage_heatmap_plot_save_path: Path key usage heatmap plot is
                                                saved to.
    """
    debug(
        "Data exploration completed",
        extra={
            "num_timestamps": len(timestamps_hours),
            "num_requests": len(requests),
            "min_key": min_key,
            "max_key": max_key,
            "data_mode": data_mode,
            "context": "Data exploration",
        },
    )

    # Prepare save paths according to data mode
    SAVE_PATHS = {
        DATA_STATIC_MODE: (
            PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH,
            PLOT_STATIC_DAILY_PROFILE_FILE_PATH,
            PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH,
        ),
        DATA_DYNAMIC_MODE: (
            PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH,
            PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH,
            PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH,
        ),
        DATA_REAL_MODE: (
            PLOT_REAL_ZIPF_LOG_LOG_FILE_PATH,
            PLOT_REAL_DAILY_PROFILE_FILE_PATH,
            PLOT_REAL_KEY_USAGE_HEATMAP_FILE_PATH,
        ),
    }
    (
        zipf_log_log_plot_save_path,
        daily_profile_plot_save_path,
        key_usage_heatmap_plot_save_path,
    ) = SAVE_PATHS.get(data_mode)

    # Explore data through plots and save them
    # via remote tasks
    plot_zipf_loglog_task.remote(requests, zipf_log_log_plot_save_path)
    plot_daily_profile_task.remote(
        timestamps_hours, daily_profile_plot_save_path
    )
    plot_key_usage_heatmap_task.remote(
        requests,
        timestamps_hours,
        min_key,
        max_key,
        key_usage_heatmap_plot_save_path,
    )

    debug(
        "Data exploration completed",
        extra={
            "context": "Data exploration",
            "plot_save_path": [
                str(zipf_log_log_plot_save_path),
                str(daily_profile_plot_save_path),
                str(key_usage_heatmap_plot_save_path),
            ],
        },
    )

    return (
        zipf_log_log_plot_save_path,
        daily_profile_plot_save_path,
        key_usage_heatmap_plot_save_path,
    )
