from const import (
    PIPELINE_PHASE_DATA_PREPROCESSING,
    REQUEST_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.data_generation.utils.dataset_saver import (
    save_dataset,
)
from lstm_eviction_policy.data_preprocessing.cleaning.duplicates_remover import (
    remove_duplicates,
)
from lstm_eviction_policy.data_preprocessing.cleaning.missing_values_remover import (
    remove_missing_values,
)
from lstm_eviction_policy.data_preprocessing.features_engineering.features_builder import (
    build_features,
)
from lstm_eviction_policy.data_preprocessing.validation.invalid_values_remover import (
    remove_invalid_values,
)
from lstm_eviction_policy.utils.data.dataset.dataset_loader import (
    load_dataset,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    info,
    phase_var,
)


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
    phase_var.set(PIPELINE_PHASE_DATA_PREPROCESSING)

    # Load the dataset
    df = load_dataset(config)

    # Remove missing values
    df = remove_missing_values(df)

    # Remove duplicates
    df = remove_duplicates(df, [TIMESTAMP_COLUMN_NAME])

    # Remove invalid values
    df = remove_invalid_values(df, config)

    # Build new features
    df = build_features(
        df,
        TIMESTAMP_COLUMN_NAME,
        REQUEST_COLUMN_NAME,
    )

    # Save preprocessed dataset
    save_dataset(df, config)

    info("Data preprocessing completed")
