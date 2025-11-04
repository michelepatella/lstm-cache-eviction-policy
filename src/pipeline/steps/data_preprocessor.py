"""data_preprocessor.py

Pipeline step module responsible for cleaning and enriching the raw
dataset generated in the previous step.

This module provides the `preprocess_data` function, which handles data
cleaning (removing missing values and duplicates) and feature engineering
(building new temporal features). The final processed dataset is saved.

Functions:
    preprocess_data() -> None
        Orchestrates the data preprocessing workflow, saving the cleaned
        and enriched dataset.
"""

import logging
import os

import dagshub
from dotenv import load_dotenv

from components.dataset.cleans.duplicates_remover import (
    remove_dataset_duplicates,
)
from components.dataset.cleans.missing_values_remover import (
    remove_dataset_missing_values,
)
from components.dataset.features.builder import (
    build_features,
)
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.io.saver import save_dataset
from components.logs.handlers.elastic_handler import ElasticHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from const import (
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
    DATASET_PROCESSED_TYPE,
    LOGS_PHASE_DATA_PREPROCESSING,
)

# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_ENV_VAR_REPO_OWNER_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_ENV_VAR_REPO_NAME)


def preprocess_data() -> None:
    """Preprocess generated data.

    This function preprocesses generated data by orchestrating missing values,
    duplicated values and invalid values removal. Finally, it orchestrates
    new features construction, saving the final preprocessed dataset for further
    usage.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_DATA_PREPROCESSING)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        dvc=DAGS_HUB_DVC,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_PHASE_DATA_PREPROCESSING,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs(logging.getLevelName(config.logs.level))

        # Prepare configuration
        data_distribution_mode = config.data.general.mode
        missing_values_removal_dropna_how = (
            config.dataset.cleaning.missing_values_removal.dropna.how
        )

        info(
            "Data preprocessing started",
            extra={
                "data_distribution_mode": data_distribution_mode,
                "context": "Data preprocessing",
            },
        )

        # Retrieve path to load dataset from
        dataset_raw_path = get_dataset_abs_path(
            DATASET_RAW_TYPE,
            data_distribution_mode,
        )

        # Load the dataset
        initial_df = load_dataset(dataset_raw_path)

        # Remove missing values
        missing_values_removed_df = remove_dataset_missing_values(
            initial_df,
            missing_values_removal_dropna_how,
        )

        # Remove duplicates
        duplicates_removed_df = remove_dataset_duplicates(
            missing_values_removed_df,
            [DATASET_COLUMN_TIMESTAMP_NAME],
        )

        # Build new features
        final_df = build_features(duplicates_removed_df)

        # Retrieve path to save dataset from
        dataset_processed_path = get_dataset_abs_path(
            DATASET_PROCESSED_TYPE,
            data_distribution_mode,
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
                "duplicates_num": len(missing_values_removed_df)
                - len(duplicates_removed_df),
                "removals_num": len(initial_df) - len(final_df),
                "removals_ratio": (len(initial_df) - len(final_df))
                / len(initial_df),
            },
        )
        mlflow.log_artifact(dataset_processed_path)

    info(
        "Data preprocessing completed",
        extra={
            "data_distribution_mode": data_distribution_mode,
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
        if isinstance(handler, ElasticHandler):
            handler.flush_buffer_sync()
