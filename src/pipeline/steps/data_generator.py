import mlflow
import numpy as np
import pandas as pd
from collections import Counter

from components.const import TIME_HOURS_IN_DAY
from components.data.requests.core.dynamic_generator import (
    generate_dynamic_requests,
)
from components.data.requests.core.static_generator import (
    generate_static_requests,
)
from components.dataset.builder import build_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.io.saver import save_dataset
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.visualization.daily_profile_plotter import (
    plot_daily_profile,
)
from components.visualization.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from components.visualization.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from src.const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_RAW_TYPE,
    DATASET_REQUEST_COLUMN_NAME,
    DATASET_TIMESTAMP_COLUMN_NAME, MLFLOW_NESTED_ENABLED,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    LOGS_DATA_GENERATION_PHASE,
    PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH,
    PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH,
    PLOT_STATIC_DAILY_PROFILE_FILE_PATH,
    PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH,
)


def generate_data() -> None:
    """
    Generate data according to a specified data distribution mode.

    This function generates data according to a specified data distribution mode,
    by orchestrating the generation of both access and temporal data patterns of
    requests. These patterns aim to reflect real-world data access patterns.
    Data generated — including a requested key and the corresponding timestamp in
    hours — is used to create a dataframe saved as CSV dataset next. Finally, data
    generated is validated by proper plots.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_DATA_GENERATION_PHASE)
    with mlflow.start_run(run_name=LOGS_DATA_GENERATION_PHASE, nested=MLFLOW_NESTED_ENABLED):

        # Setup
        config = prepare_config()
        initialize_logs()

        info("Data generation started")

        # Prepare configuration
        data_distribution_mode = config.data.mode
        min_key = config.data.keys.min
        max_key = config.data.keys.max

        # Generate requests with corresponding timestamps,
        # based on the data distribution mode
        if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
            # Static requests generation
            requests, timestamps_hours = generate_static_requests(config)

            # Prepare static save paths
            zipf_log_log_plot_save_path = PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH
            daily_profile_plot_save_path = PLOT_STATIC_DAILY_PROFILE_FILE_PATH
            key_usage_heatmap_plot_save_path = (
                PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH
            )
        else:
            # Dynamic requests generation
            requests, timestamps_hours = generate_dynamic_requests(config)

            # Prepare dynamic save paths
            zipf_log_log_plot_save_path = PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH
            daily_profile_plot_save_path = PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH
            key_usage_heatmap_plot_save_path = (
                PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH
            )

        # Create a dataset where each row is composed of
        # a timestamp and the corresponding request
        df = build_dataset(
            {
                DATASET_TIMESTAMP_COLUMN_NAME: timestamps_hours[: len(requests)],
                DATASET_REQUEST_COLUMN_NAME: requests,
            }
        )

        # Retrieve path where
        # to save dataset
        dataset_path = get_dataset_abs_path(
            DATASET_RAW_TYPE, data_distribution_mode
        )

        # Save just created dataset
        save_dataset(df, dataset_path)

        # Show data generation -related plots
        plot_zipf_loglog(requests, zipf_log_log_plot_save_path)
        plot_daily_profile(timestamps_hours, daily_profile_plot_save_path)
        plot_key_usage_heatmap(
            min_key,
            max_key,
            requests,
            timestamps_hours,
            key_usage_heatmap_plot_save_path,
        )

        mlflow.log_metrics(
            {
                "dataset_num_rows": len(df),
                "dataset_num_columns": len(df.columns),
                "requests_num_unique": len(np.unique(requests)),
                "requests_max_count": max(Counter(requests)),
                "requests_min_count": min(Counter(requests)),
                "requests_mean": float(np.mean(requests)),
                "requests_std": float(np.std(requests)),
                "requests_skew": float(pd.Series(requests).skew()),
                "requests_kurt": float(pd.Series(requests).kurt()),
                "timestamps_min": float(min(timestamps_hours)),
                "timestamps_max": float(max(timestamps_hours)),
                "timestamps_mean": float(np.mean(timestamps_hours)),
                "timestamps_std": float(np.std(timestamps_hours)),
                "timestamps_diff_mean": float(np.mean(np.diff(timestamps_hours))),
                "timestamps_diff_std": float(np.std(np.diff(timestamps_hours))),
                "timestamps_diff_min": float(np.min(np.diff(timestamps_hours))),
                "timestamps_diff_max": float(np.max(np.diff(timestamps_hours))),
                "total_hours": max(timestamps_hours) - min(timestamps_hours),
                "total_days": (max(timestamps_hours) - min(timestamps_hours))
                / TIME_HOURS_IN_DAY,
            }
        )
        mlflow.log_artifact(dataset_path)
        mlflow.log_artifact(zipf_log_log_plot_save_path)
        mlflow.log_artifact(daily_profile_plot_save_path)
        mlflow.log_artifact(key_usage_heatmap_plot_save_path)
        mlflow.end_run()

    info("Data generation completed")


if __name__ == "__main__":
    generate_data()
