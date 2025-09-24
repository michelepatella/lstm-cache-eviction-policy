import pandas as pd

from lstm_eviction_policy.data_preprocessing.utils.dataset_cleaning_reporter import (
    report_dataset_cleaning,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    error,
    info,
)


def remove_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove missing values from dataset.

    This function removes rows with missing values
    from dataset, returning a new clean dataset.

    Parameters:
        df (pd.DataFrame): Dataset to remove missing
                           values from.

    Returns:
        pd.DataFrame: Dataset without missing values.

    Raises:
        AttributeError: If dataframe to be cleaned is
                        not a pd.DataFrame.
    """
    try:
        # Remove rows with missing values
        new_df = df.dropna(axis=0, how="any")

        # For missing values report
        report_dataset_cleaning(df, new_df, "Missing values removal")

        info("Missing values removal completed")

        return new_df
    except AttributeError as e:
        msg = "Failed to remove missing values from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
