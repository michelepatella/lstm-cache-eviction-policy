import pandas as pd

from const import (
    REQUEST_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.data_preprocessing.utils.dataset_cleaning_reporter import (
    report_dataset_cleaning,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    error,
    info,
)


def remove_invalid_values(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Remove invalid values from dataset.

    This function removes rows with invalid column values
    from dataset, returning a new valid dataset.

    Parameters:
        df (pd.DataFrame): Dataset to remove invalid
                           values from.
        config (Config): Configuration object.

    Returns:
        pd.DataFrame: Dataset without invalid values.

    Raises:
        RuntimeError: If an error occurs while removing invalid values from dataset, e.g.:
            * If dataset to be validated is not a Pandas DataFrame.
            * If columns to validate do not exist in the dataset.
            * If dataset is not iterable or of incompatible type.
    """
    try:
        # Retrieve min and max keys
        min_key = config.data.general.keys.min
        max_key = config.data.general.keys.max

        # Filter out rows having valid
        # request values only
        valid_request_rows = df[REQUEST_COLUMN_NAME].apply(
            lambda x: isinstance(x, int) and min_key <= x <= max_key
        )
        new_df = df[valid_request_rows]

        # Filter out rows having valid
        # timestamp values only
        valid_timestamp_rows = pd.to_numeric(
            df[TIMESTAMP_COLUMN_NAME],
            errors="coerce",
        ).notna()
        new_df = new_df[valid_timestamp_rows]

        # Report for invalid values removal
        report_dataset_cleaning(df, new_df, "Invalid values removal")

        info("Invalid values removal completed")

        return new_df
    except (
        AttributeError,
        KeyError,
        TypeError,
    ) as e:
        msg = "Failed to remove invalid values from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
