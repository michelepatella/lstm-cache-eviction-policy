from const import (
    DATASET_PREPROCESSED_TYPE,
    DATASET_RAW_TYPE,
    LOGS_DATA_PREPROCESSING_PHASE,
    TIMESTAMP_COLUMN_NAME,
)
from pipeline.config.configurator import prepare_config
from pipeline.steps.data_preprocessing.features.builder import (
    build_time_features,
)
from pipeline.utils.dataset.cleaning.duplicates_remover import (
    remove_dataset_duplicates,
)
from pipeline.utils.dataset.cleaning.missing_values_remover import (
    remove_dataset_missing_values,
)
from pipeline.utils.dataset.saver import save_dataset
from pipeline.utils.dataset.loader import load_dataset
from pipeline.utils.logs.initializer import logs_phase
from pipeline.utils.logs.levels.info_logger import info


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

    # Load the dataset
    df = load_dataset(DATASET_RAW_TYPE, config)

    # Remove missing values
    df = remove_dataset_missing_values(df)

    # Remove duplicates
    df = remove_dataset_duplicates(df, [TIMESTAMP_COLUMN_NAME])

    # Build new features
    df = build_time_features(df)

    # Save preprocessed dataset
    save_dataset(df, DATASET_PREPROCESSED_TYPE, config)

    info("Data preprocessing completed")


if __name__ == "__main__":
    preprocess_data()
