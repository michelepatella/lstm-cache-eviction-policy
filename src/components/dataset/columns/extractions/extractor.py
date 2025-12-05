"""extractor.py

Utility module for extracting column names from pandas DataFrames.

This module provides the `extract_dataset_columns` function, which returns
the list of column names from a given DataFrame.

Functions:
    extract_dataset_columns(df: pd.DataFrame) -> list[str]
        Extracts and returns column names from the provided DataFrame.
"""

import pandas as pd

from components.logs.levels.error_logger import error


def extract_dataset_columns(df: pd.DataFrame) -> list[str]:
    """Extract columns from a dataset.

    This function extracts and returns a list of column names
    belonging to the provided dataset.

    Args:
        df (pd.DataFrame): Dataset to extract columns from.

    Returns:
        list[str]: List of column names extracted.

    Raises:
        RuntimeError: If extracting columns fails:
            * Dataset does not have columns attribute (AttributeError).
            * Columns attribute is not iterable (TypeError).
    """
    try:
        # Extract column names
        return df.columns.tolist()
    except (AttributeError, TypeError) as e:
        msg = "Dataset columns extraction from dataset failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "df_type": type(df).__name__,
                "has_columns_attr": hasattr(df, "columns"),
                "context": "Dataset columns extraction from dataset",
            },
        )
        raise RuntimeError(msg) from e
