"""data_preparer.py

Module responsible for the data preparation phase of the pipeline.

This module orchestrates the workflow for generating, loading, saving, exploring,
and tracking the raw dataset.

Functions:
    prepare_data() -> None
        Main function to execute the data preparation pipeline step.
"""

import logging
from collections import Counter

import dagshub
import mlflow
import numpy as np
import pandas as pd

from components.data.exploration.explorer import explore_data
from components.data.requests.core.dynamic_generator import (
    generate_dynamic_requests,
)
from components.data.requests.core.static_generator import (
    generate_static_requests,
)
from components.dataset.builder import build_dataset
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.io.saver import save_dataset
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.ray.initializer import initialize_ray
from components.seed.setter import set_seed
from const import (
    DATA_REAL_MODE,
    DATA_STATIC_MODE,
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
    DATASET_RAW_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import (
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    LOGS_PHASE_DATA_PREPARATION,
)


def prepare_data() -> None:
    """Executes the data preparation phase of the pipeline.

    This function first determines whether to generate synthetic requests
    (using static or dynamic modes) or load an existing real dataset. It then
    assembles the timestamps and requests into a raw DataFrame and saves it
    to the designated location. Finally, it explores the data's characteristics,
    calculating various statistical metrics and generating analytical plots (like
    Zipf distribution and key usage heatmaps) to visualize the dataset properties.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_DATA_PREPARATION)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        mlflow=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_DATA_PREPARATION,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        pipeline_config = prepare_pipeline_config()
        initialize_logs(
            logging.getLevelName(pipeline_config.logs.level),
            GrafanaLokiHandler(),
        )
        initialize_ray(
            pipeline_config.resources.general.num_cpus,
            pipeline_config.resources.general.num_gpus,
        )

        # Prepare configuration
        data_mode = pipeline_config.data.general.mode
        min_key = pipeline_config.data.general.keys.min
        max_key = pipeline_config.data.general.keys.max
        seed = pipeline_config.seed.value

        # Ensure reproducibility
        set_seed(seed)

        # Retrieve dataset path for further usage
        dataset_path = get_dataset_abs_path(
            DATASET_RAW_TYPE,
            data_mode,
        )

        info(
            "Data preparation started",
            extra={
                "data_mode": data_mode,
                "dataset_path": str(dataset_path),
                "key_min": min_key,
                "key_max": max_key,
                "context": "Data preparation",
            },
        )

        # Check whether data needs to be
        # synthetically generated
        if data_mode != DATA_REAL_MODE:
            # Generate requests with corresponding timestamps,
            # based on the data distribution mode
            if data_mode == DATA_STATIC_MODE:
                # Static requests generation
                requests, timestamps_hours = generate_static_requests(
                    pipeline_config,
                )
            else:
                # Dynamic requests generation
                requests, timestamps_hours = generate_dynamic_requests(
                    pipeline_config,
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

            # Save just created dataset
            save_dataset(df, dataset_path)
        else:
            # Dataset already exists, just load it
            df = load_dataset(dataset_path)

            # Extract columns (timestamps and requests)
            timestamps_hours = df[DATASET_COLUMN_TIMESTAMP_NAME]
            requests = df[DATASET_COLUMN_REQUEST_NAME]

        # Explore data
        (
            zipf_log_log_plot_save_path,
            daily_profile_plot_save_path,
            key_usage_heatmap_plot_save_path,
        ) = explore_data(
            timestamps_hours,
            requests,
            min_key,
            max_key,
            data_mode,
        )

        # Experiment tracking
        mlflow.log_params(prepare_pipeline_config().model_dump())
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
        "Data preparation completed",
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
            "context": "Data preparation",
        },
    )


if __name__ == "__main__":
    prepare_data()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
