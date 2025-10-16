import pandas as pd

from const import (
    MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    MISSING_VALUES_REMOVAL_DROPNA_HOW,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def remove_dataset_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove missing values from dataset.

    This function removes rows with missing values
    from dataset, returning a new clean dataset.

    Args:
        df (pd.DataFrame): Dataset to remove missing
                           values from.

    Returns:
        pd.DataFrame: Dataset without missing values.

    Raises:
        RuntimeError: If dataframe to be cleaned is
                      not a pd.DataFrame.
    """
    try:
        # Remove rows with missing values
        new_df = df.dropna(
            axis=MISSING_VALUES_REMOVAL_DROPNA_AXIS,
            how=MISSING_VALUES_REMOVAL_DROPNA_HOW,
        )

        info("Dataset missing values removal completed")

        return new_df
    except AttributeError as e:
        msg = "Failed to remove missing values from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
