"""data_generator.py

Pipeline step module responsible for generating synthetic dataset requests
and temporal data based on specified distribution modes.

This module provides the `generate_data` function, which orchestrates the
creation of raw request data (static or dynamic), builds it into a DataFrame
dataset, saves the raw dataset, and validates the generated data distribution
through various plots (Zipf, daily profile, heatmap).

Functions:
    generate_data() -> None
        Orchestrates the data generation process, saving the raw dataset
        and related validation plots.
"""

import logging
import os
from collections import Counter

import dagshub
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from components.data.requests.core.dynamic_generator import (
    generate_dynamic_requests,
)
from components.data.requests.core.static_generator import (
    generate_static_requests,
)
from components.dataset.builder import build_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.io.saver import save_dataset
from components.logs.handlers.elastic_handler import ElasticHandler
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
from const import (
    DATA_STATIC_MODE,
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
    DATASET_RAW_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    DAGS_HUB_DVC,
    DAGS_HUB_ENV_VAR_REPO_NAME,
    DAGS_HUB_ENV_VAR_REPO_OWNER_NAME,
    LOGS_PHASE_DATA_GENERATION,
    PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH,
    PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH,
    PLOT_STATIC_DAILY_PROFILE_FILE_PATH,
    PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH,
    PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH,
)

# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_ENV_VAR_REPO_OWNER_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_ENV_VAR_REPO_NAME)


def generate_data() -> None:
    """Generate data according to a specified data distribution mode.

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
    logs_phase.set(LOGS_PHASE_DATA_GENERATION)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        mlflow=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_DATA_GENERATION,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs(logging.getLevelName(config.logs.level))

        # Prepare configuration
        data_mode = config.data.general.mode
        min_key = config.data.general.keys.min
        max_key = config.data.general.keys.max

        info(
            "Data generation started",
            extra={
                "data_mode": data_mode,
                "key_min": min_key,
                "key_max": max_key,
                "context": "Data generation",
            },
        )

        # Generate requests with corresponding timestamps,
        # based on the data distribution mode
        if data_mode == DATA_STATIC_MODE:
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
                DATASET_COLUMN_TIMESTAMP_NAME: timestamps_hours[
                    : len(requests)
                ],
                DATASET_COLUMN_REQUEST_NAME: requests,
            },
        )

        # Retrieve path where
        # to save dataset
        dataset_path = get_dataset_abs_path(
            DATASET_RAW_TYPE,
            data_mode,
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

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_metrics(
            {
                "dataset_rows_num": len(df),
                "dataset_columns_num": len(df.columns),
                "requests_unique_num": len(np.unique(requests)),
                "requests_max_num": max(Counter(requests)),
                "requests_min_num": min(Counter(requests)),
                "requests_mean": float(np.mean(requests)),
                "requests_std": float(np.std(requests)),
                "requests_skew": float(pd.Series(requests).skew()),
                "requests_kurt": float(pd.Series(requests).kurt()),
                "timestamps_hours_min": float(min(timestamps_hours)),
                "timestamps_hours_max": float(max(timestamps_hours)),
                "timestamps_hours_mean": float(np.mean(timestamps_hours)),
                "timestamps_hours_std": float(np.std(timestamps_hours)),
                "timestamps_hours_diff_mean": float(
                    np.mean(np.diff(timestamps_hours)),
                ),
                "timestamps_hours_diff_std": float(
                    np.std(np.diff(timestamps_hours)),
                ),
                "timestamps_hours_diff_min": float(
                    np.min(np.diff(timestamps_hours)),
                ),
                "timestamps_hours_diff_max": float(
                    np.max(np.diff(timestamps_hours)),
                ),
                "days_num": 1
                + sum(
                    timestamps_hours[i] < timestamps_hours[i - 1]
                    for i in range(1, len(timestamps_hours))
                ),
            },
        )
        mlflow.log_artifact(dataset_path)
        mlflow.log_artifact(zipf_log_log_plot_save_path)
        mlflow.log_artifact(daily_profile_plot_save_path)
        mlflow.log_artifact(key_usage_heatmap_plot_save_path)

    info(
        "Data generation completed",
        extra={
            "data_mode": data_mode,
            "dataset_raw_save_path": str(dataset_path),
            "rows_num": len(df),
            "columns_num": len(df.columns),
            "unique_requests_num": len(set(requests)),
            "plot_save_path": [
                str(zipf_log_log_plot_save_path),
                str(daily_profile_plot_save_path),
                str(key_usage_heatmap_plot_save_path),
            ],
            "context": "Data generation",
        },
    )


if __name__ == "__main__":
    generate_data()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, ElasticHandler):
            handler.flush_buffer_sync()
