from typing import Union

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def shift_dataset_column(
    df: pd.DataFrame, column_name: str, shift: Union[int, float]
) -> None:
    """
    Shift the values of a dataset column.

    This function applies the given shift and updates the
    dataset in-place.

    Args:
        df (pd.DataFrame): Dataset containing the column to shift.
        column_name (str): Name of the column to shift.
        shift (Union[int, float]): Value to add/subtract from column values.

    Returns:
        None

    Raises:
        RuntimeError: If shifting the column fails:
            * Column not found in dataset (KeyError).
            * Dataset or column values not suitable for arithmetic
              (TypeError, AttributeError).
    """
    try:
        debug(f"Dataset column to shift: {column_name}")
        debug(f"Shift to apply to dataset column: {shift}")

        # Cast to int and apply shift to column
        df[column_name] = df[column_name] + shift

        debug("Dataset column shifted")
    except (TypeError, AttributeError, KeyError) as e:
        msg = "Failed to shift dataset column"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
