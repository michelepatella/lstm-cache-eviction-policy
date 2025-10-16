from typing import List

import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def extract_dataset_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract columns from a dataset.

    This function extracts and returns a list of
    column names belonging to the provided dataset.

    Args:
        df (pd.DataFrame): Dataset to extract columns from.

    Returns:
        List[str]: List of column names extracted.

    Raises:
        RuntimeError: If an error occurs while extracting dataset columns e.g.:
            * If data is not a DataFrame.
            * If data has no columns.
    """
    try:
        # Extract column names
        columns = df.columns.tolist()
        debug(f"Columns extracted from dataset: {columns}")
    except (AttributeError, TypeError) as e:
        msg = "Failed to extract columns from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Columns from dataset extraction completed")

    return columns
