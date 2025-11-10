"""dropper.py

Utility module for dropping columns from a dataset.

This module provides the `drop_dataset_column` function, which removes a
specified column from a pandas DataFrame.

Functions:
    drop_dataset_column(
        df: pd.DataFrame,
        column_name: str
    ) -> pd.DataFrame
        Drops the specified column from the provided DataFrame.
"""

import pandas as pd

from components.logs.levels.error_logger import error


def drop_dataset_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drop a column from the dataset.

    This function, given a DataFrame and a column name, removes the
    column from the dataset.

    Args:
        df (pd.DataFrame): Dataset to update.
        column_name (str): Name of the column to drop.

    Returns:
        pd.DataFrame: DataFrame with the column removed.

    Raises:
        RuntimeError: If dropping the column fails:
            * Column does not exist in the dataset (KeyError).
    """
    try:
        # Drop column from dataset
        return df.drop(columns=[column_name])
    except KeyError as e:
        msg = "Column removal from dataset failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "column_requested": column_name,
                "columns_available": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "context": "Column removal from dataset",
            },
        )
        raise RuntimeError(msg) from e
