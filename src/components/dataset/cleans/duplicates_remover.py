from typing import List

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def remove_dataset_duplicates(
    df: pd.DataFrame, subset: List[str]
) -> pd.DataFrame:
    """
    Remove duplicates from dataset.

    This function removes rows with duplicated values from a given
    subset of dataset column(s), returning a new clean dataset.

    Args:
        df (pd.DataFrame): Dataset to remove duplicates from.
        subset (List[str]): Column(s) of dataset to remove duplicates from.

    Returns:
        pd.DataFrame: Dataset without duplicated values.

     Raises:
        RuntimeError: If removing duplicates fails:
            * Dataset object is not a valid DataFrame (AttributeError).
            * Specified subset columns are invalid (KeyError, TypeError).
    """
    try:
        # Remove duplicates from dataset
        new_df = df.drop_duplicates(subset=subset)

        debug("Dataset duplicates removal completed")

        return new_df
    except (AttributeError, KeyError, TypeError) as e:
        msg = "Failed to remove duplicates from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
