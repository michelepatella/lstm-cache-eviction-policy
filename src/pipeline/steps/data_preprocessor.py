"""data_preprocessor.py

Pipeline step module responsible for cleaning and enriching the raw
dataset.

This module provides the `preprocess_data` function, which handles data
cleaning (removing missing values) and feature engineering
(building new temporal features). The final processed dataset is saved.

Functions:
    preprocess_data() -> None
        Orchestrates the data preprocessing workflow, saving the cleaned
        and enriched dataset.
"""

import logging

import dagshub
import mlflow
import numpy as np
import pandas as pd
import ray

from components.const import (
    DATASET_INDEX,
    LIST_FIRST_IDX,
)
from components.dataset.cleans.missing_values_remover import (
    remove_dataset_missing_values,
)
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.io.saver import save_dataset
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.ray.initializer import initialize_ray
from components.ray.tasks.features.builder import build_features_task
from components.seed.setter import set_seed
from const import (
    DATASET_RAW_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    DATASET_PROCESSED_TYPE,
    DATASET_RESET_INDEX_DROP,
    LOGS_PHASE_DATA_PREPROCESSING,
)


def preprocess_data() -> None:
    """Preprocess data.

    This function preprocesses data by orchestrating missing values removal and
    new features construction, saving the final preprocessed dataset for further
    usage.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_DATA_PREPROCESSING)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        dvc=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_DATA_PREPROCESSING,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs(
            logging.getLevelName(config.logs.level), GrafanaLokiHandler()
        )
        initialize_ray(
            config.resources.general.num_cpus,
            config.resources.general.num_gpus,
        )

        # Prepare configuration
        data_mode = config.data.general.mode
        missing_values_removal_dropna_how = (
            config.dataset.cleaning.missing_values_removal.dropna.how
        )
        seq_len = config.model.sequence.length
        num_cpus = config.resources.general.num_cpus
        num_gpus = config.resources.general.num_gpus
        seed = config.seed.value

        # Ensure reproducibility
        set_seed(seed)

        info(
            "Data preprocessing started",
            extra={
                "data_mode": data_mode,
                "missing_values_removal_dropna_how": missing_values_removal_dropna_how,
                "seq_len": seq_len,
                "context": "Data preprocessing",
            },
        )

        # Retrieve path to load dataset from
        dataset_raw_path = get_dataset_abs_path(
            DATASET_RAW_TYPE,
            data_mode,
        )

        # Load the dataset
        initial_df = load_dataset(dataset_raw_path)

        # Remove missing values
        missing_values_removed_df = remove_dataset_missing_values(
            initial_df,
            missing_values_removal_dropna_how,
        )

        # Determine the dataset chunks dimension as the
        # initial length divided by the max among CPUs
        # and GPUs available
        df_chunk_size = int(
            np.ceil(len(initial_df) / max(num_cpus, num_gpus)),
        )

        # Create dataset chunks with overlap of last sequence
        # length requests
        df_chunks = [
            initial_df.iloc[
                max(LIST_FIRST_IDX, i * df_chunk_size - seq_len) : min(
                    len(initial_df),
                    (i + 1) * df_chunk_size,
                )
            ].copy()
            for i in range(max(num_cpus, num_gpus))
        ]

        # Build features in a distributed way
        # via remote Ray tasks, each one of them
        # working on a dataset chunk
        futures = [
            build_features_task.remote(
                chunk.reset_index(drop=DATASET_RESET_INDEX_DROP),
                seq_len,
            )
            for chunk in df_chunks
        ]
        processed_chunks = ray.get(futures)

        # Build final dataset by removing overlapping
        # requests and concatenating preprocessed dataset chunks
        final_chunks = [
            chunk_df if i == LIST_FIRST_IDX else chunk_df.iloc[seq_len:].copy()
            for i, chunk_df in enumerate(processed_chunks)
        ]
        final_df = pd.concat(final_chunks, ignore_index=DATASET_INDEX)

        # Retrieve path to save dataset from
        dataset_processed_path = get_dataset_abs_path(
            DATASET_PROCESSED_TYPE,
            data_mode,
        )

        # Save preprocessed dataset
        save_dataset(final_df, dataset_processed_path)

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_metrics(
            {
                "dataset_rows_num": len(final_df),
                "dataset_columns_num": len(final_df.columns),
                "missing_values_num": len(initial_df)
                - len(missing_values_removed_df),
            },
        )
        mlflow.log_artifact(dataset_processed_path)

    info(
        "Data preprocessing completed",
        extra={
            "data_mode": data_mode,
            "dataset_raw_save_path": str(dataset_raw_path),
            "dataset_processed_save_path": str(dataset_processed_path),
            "rows_final_num": len(final_df),
            "columns_final_num": len(final_df.columns),
            "context": "Data preprocessing",
        },
    )


if __name__ == "__main__":
    preprocess_data()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
