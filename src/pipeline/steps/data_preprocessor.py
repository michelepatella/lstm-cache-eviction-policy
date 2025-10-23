import dagshub

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
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    DATASET_PROCESSED_TYPE,
    LOGS_DATA_PREPROCESSING_PHASE,
)
from src.const import (
    DAGS_HUB_MLFLOW_ENABLED,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    DATASET_RAW_TYPE,
    DATASET_TIMESTAMP_COLUMN_NAME,
    MLFLOW_NESTED_ENABLED,
)


def preprocess_data() -> None:
    """
    Preprocess generated data.

    This function preprocesses generated data by orchestrating missing values,
    duplicated values and invalid values removal. Finally, it orchestrates
    new features construction, saving the final preprocessed dataset for further
    usage.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_DATA_PREPROCESSING_PHASE)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_DATA_PREPROCESSING_PHASE, nested=MLFLOW_NESTED_ENABLED
    ):

        # Setup
        config = prepare_config()
        initialize_logs()

        info("Data preprocessing started")

        # Prepare configuration
        data_distribution_mode = config.data.mode

        # Retrieve path to load dataset from
        dataset_raw_path = get_dataset_abs_path(
            DATASET_RAW_TYPE, data_distribution_mode
        )

        # Load the dataset
        initial_df = load_dataset(dataset_raw_path)

        # Remove missing values
        missing_values_removed_df = remove_dataset_missing_values(initial_df)

        # Remove duplicates
        duplicates_removed_df = remove_dataset_duplicates(
            missing_values_removed_df, [DATASET_TIMESTAMP_COLUMN_NAME]
        )

        # Build new features
        final_df = build_features(duplicates_removed_df)

        # Retrieve path to save dataset from
        dataset_processed_path = get_dataset_abs_path(
            DATASET_PROCESSED_TYPE, data_distribution_mode
        )

        # Save preprocessed dataset
        save_dataset(final_df, dataset_processed_path)

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_metrics(
            {
                "dataset_num_rows": len(final_df),
                "dataset_num_columns": len(final_df.columns),
                "missing_values_num": len(initial_df)
                - len(missing_values_removed_df),
                "duplicates_num": len(missing_values_removed_df)
                - len(duplicates_removed_df),
                "removals_total": len(initial_df) - len(final_df),
                "removals_ratio": (len(initial_df) - len(final_df))
                / len(initial_df),
            }
        )
        mlflow.log_artifact(dataset_processed_path)
        mlflow.end_run()

    info("Data preprocessing completed")


if __name__ == "__main__":
    preprocess_data()
