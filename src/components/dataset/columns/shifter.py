import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def shift_dataset_column(
    df: pd.DataFrame, column_name: str, shift: int | float
) -> None:
    """
    Shift the values of a dataset column.

    This function applies the given shift and updates
    the dataset in-place.

    Args:
        df (pd.DataFrame): Dataset containing the column to shift.
        column_name (str): Name of the column to shift.
        shift (int | float): Value to add/subtract from column values.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while shifting the dataset column, e.g.:
            * Dataset column does not exist.
            * Dataset is not valid.
    """
    debug(f"Dataset column to shift: {column_name}")
    debug(f"Shift to apply to dataset column: {shift}")

    try:
        # Cast to int and apply shift to column
        df[column_name] = df[column_name] + shift
    except (TypeError, AttributeError, KeyError) as e:
        msg = "Failed to shift dataset column"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Dataset column shifted")
