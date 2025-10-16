from pipeline.config.configurator import prepare_config
from pipeline.const import (
    DATASET_RAW_TYPE,
    LOGS_DATA_PREPROCESSING_PHASE,
    TIMESTAMP_COLUMN_NAME,
)
from components.dataset.features.builder import (
    build_features,
)
from features.dataset.cleaning.duplicates_remover import (
    remove_dataset_duplicates,
)
from features.dataset.cleaning.missing_values_remover import (
    remove_dataset_missing_values,
)
from features.dataset.io.loader import load_dataset
from features.dataset.io.locator import get_dataset_abs_path
from features.dataset.io.saver import save_dataset
from features.logs.initializer import logs_phase
from features.logs.levels.info_logger import info


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

    # Read configuration
    config = prepare_config()

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode

    # Retrieve path to load dataset from
    dataset_path = get_dataset_abs_path(
        DATASET_RAW_TYPE, data_distribution_mode
    )

    # Load the dataset
    df = load_dataset(dataset_path)

    # Remove missing values
    df = remove_dataset_missing_values(df)

    # Remove duplicates
    df = remove_dataset_duplicates(df, [TIMESTAMP_COLUMN_NAME])

    # Build new features
    df = build_features(df)

    # Save preprocessed dataset
    save_dataset(df, dataset_path)

    info("Data preprocessing completed")


if __name__ == "__main__":
    preprocess_data()
