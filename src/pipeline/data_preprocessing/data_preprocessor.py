from const import (
    DATASET_PREPROCESSED_TYPE,
    DATASET_RAW_TYPE,
    LOGS_DATA_PREPROCESSING_PHASE,
    REQUEST_COLUMN,
    TIMESTAMP_COLUMN,
)
from pipeline.config.classes.Config import Config
from pipeline.data_preprocessing.features.builder import (
    build_features,
)
from pipeline.data_preprocessing.removal.duplicates_remover import (
    remove_duplicates,
)
from pipeline.data_preprocessing.removal.invalid_values_remover import (
    remove_invalid_values,
)
from pipeline.data_preprocessing.removal.missing_values_remover import (
    remove_missing_values,
)
from pipeline.utils.dataset.saver import save_dataset
from utils.dataset.loader import load_dataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info


def preprocess_data(config: Config) -> None:
    """
    Preprocess generated data.

    This function preprocesses generated data by
    orchestrating missing values, duplicated values
    and invalid values removal. Finally, it orchestrates
    new features construction, saving the final
    preprocessed dataset for further usage.

    Parameters:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set the new pipeline state
    logs_phase.set(LOGS_DATA_PREPROCESSING_PHASE)

    # Load the dataset
    df = load_dataset(DATASET_RAW_TYPE, config)

    # Remove missing values
    df = remove_missing_values(df)

    # Remove duplicates
    df = remove_duplicates(df, [TIMESTAMP_COLUMN])

    # Remove invalid values
    df = remove_invalid_values(df, config)

    # Build new features
    df = build_features(
        df,
        TIMESTAMP_COLUMN,
        REQUEST_COLUMN,
    )

    # Save preprocessed dataset
    save_dataset(df, DATASET_PREPROCESSED_TYPE, config)

    info("Data preprocessing completed")
