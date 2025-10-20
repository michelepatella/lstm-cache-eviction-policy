from typing import Literal

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import (
    MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    MISSING_VALUES_REMOVAL_DROPNA_HOW,
)


def remove_dataset_missing_values(
    df: pd.DataFrame,
    dropna_axis: int = MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    dropna_how: Literal["any", "all"] = MISSING_VALUES_REMOVAL_DROPNA_HOW,
) -> pd.DataFrame:
    """
    Remove missing values from dataset.

    This function removes rows with missing values from dataset,
    returning a new clean dataset.

    Args:
        df (pd.DataFrame): Dataset to remove missing values from.
        dropna_axis (int): Axis along which to remove missing values
                           (0 for rows, 1 for columns).
        dropna_how (Literal["any", "all"]): Determines if a row/column is removed
                                            when any or all values are missing.

    Returns:
        pd.DataFrame: Dataset without missing values.

    Raises:
        RuntimeError: If removing missing values fails:
            * Dataset does not have expected attributes (AttributeError).
            * Invalid axis or how argument for pandas dropna (TypeError, ValueError).
    """
    try:
        # Remove rows with missing values
        new_df = df.dropna(
            axis=dropna_axis,
            how=dropna_how,
        )

        debug("Dataset missing values removal completed")

        return new_df
    except (AttributeError, TypeError, ValueError) as e:
        msg = "Failed to remove missing values from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
