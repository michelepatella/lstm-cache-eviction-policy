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
from const import (
    LOGS_DATA_PREPROCESSING_PHASE,
    TIMESTAMP_COLUMN_NAME, DATASET_PROCESSED_TYPE, DATASET_RAW_TYPE,
)
from pipeline.config.configurator import prepare_config


def preprocess_data() -> None:
    """
    Preprocess generated data.

    This function preprocesses generated data by
    orchestrating missing values, duplicated values
    and invalid values removal. Finally, it orchestrates
    new features construction, saving the final
    preprocessed dataset for further usage.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_DATA_PREPROCESSING_PHASE)

    # Setup
    config = prepare_config()
    initialize_logs()

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode

    # Retrieve path to load dataset from
    dataset_raw_path = get_dataset_abs_path(
        DATASET_RAW_TYPE, data_distribution_mode
    )

    # Load the dataset
    df = load_dataset(dataset_raw_path)

    # Remove missing values
    df = remove_dataset_missing_values(df)

    # Remove duplicates
    df = remove_dataset_duplicates(df, [TIMESTAMP_COLUMN_NAME])

    # Build new features
    df = build_features(df)

    # Retrieve path to save dataset from
    dataset_processed_path = get_dataset_abs_path(
        DATASET_PROCESSED_TYPE, data_distribution_mode
    )

    # Save preprocessed dataset
    save_dataset(df, dataset_processed_path)

    info("Data preprocessing completed")


if __name__ == "__main__":
    preprocess_data()
