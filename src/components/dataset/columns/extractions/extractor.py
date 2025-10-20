from typing import List

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def extract_dataset_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract columns from a dataset.

    This function extracts and returns a list of column names
    belonging to the provided dataset.

    Args:
        df (pd.DataFrame): Dataset to extract columns from.

    Returns:
        List[str]: List of column names extracted.

    Raises:
        RuntimeError: If extracting columns fails:
            * Dataset does not have columns attribute (AttributeError).
            * Columns attribute is not iterable (TypeError).
    """
    try:
        # Extract column names
        columns = df.columns.tolist()

        debug(f"{columns} columns extracted from dataset")

        return columns
    except (AttributeError, TypeError) as e:
        msg = "Failed to extract columns from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
