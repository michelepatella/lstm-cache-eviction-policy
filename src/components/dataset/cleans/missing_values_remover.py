"""missing_values_remover.py

Utility module for removing missing values from pandas DataFrames.

This module provides the `remove_dataset_missing_values` function, which
removes rows or columns containing missing values (`NaN`) from a DataFrame.
It supports customization of the axis and the removal strategy (any or all
missing values).

Functions:
    remove_dataset_missing_values(
        df: DataFrame,
        dropna_how: str,
        dropna_axis: int = MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    ) -> DataFrame
        Removes missing values from the dataset according to the specified
        axis and strategy.
"""

import pandas as pd

from components.const import (
    DATASET_MISSING_VALUES_REMOVAL_DROPNA_AXIS,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def remove_dataset_missing_values(
    df: pd.DataFrame,
    dropna_how: str,
    dropna_axis: int = DATASET_MISSING_VALUES_REMOVAL_DROPNA_AXIS,
) -> pd.DataFrame:
    """Remove missing values from dataset.

    This function removes rows with missing values from dataset,
    returning a new clean dataset.

    Args:
        df (pd.DataFrame): Dataset to remove missing values from.
        dropna_how (str): Determines if a row/column is removed when
                          any or all values are missing.
        dropna_axis (int): Axis along which to remove missing values
                           (0 for rows, 1 for columns).

    Returns:
        pd.DataFrame: Dataset without missing values.

    Raises:
        RuntimeError: If removing missing values fails:
            * Dataset does not have expected attributes (AttributeError).
            * Invalid axis or how argument for pandas dropna
              (TypeError, ValueError).
    """
    try:
        debug(
            "Dataset missing values removal started",
            extra={
                "rows_before_num": (
                    len(df) if isinstance(df, pd.DataFrame) else None
                ),
                "columns_before_num": (
                    len(df.columns) if isinstance(df, pd.DataFrame) else None
                ),
                "dropna_axis": dropna_axis,
                "dropna_how": dropna_how,
                "context": "Dataset missing values removal",
            },
        )

        # Remove rows with missing values
        new_df = df.dropna(
            axis=dropna_axis,
            how=dropna_how,
        )

        debug(
            "Dataset missing values removal completed",
            extra={
                "rows_after_num": len(new_df),
                "columns_after_num": len(new_df.columns),
                "rows_removed_num": (
                    len(df) - len(new_df) if dropna_axis == 0 else None
                ),
                "columns_removed_num": (
                    len(df.columns) - len(new_df.columns)
                    if dropna_axis == 1
                    else None
                ),
                "dropna_axis": dropna_axis,
                "dropna_how": dropna_how,
                "context": "Dataset missing values removal",
            },
        )

        return new_df
    except (AttributeError, TypeError, ValueError) as e:
        msg = "Dataset missing values removal failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dropna_axis": dropna_axis,
                "dropna_how": dropna_how,
                "rows_num": len(df) if isinstance(df, pd.DataFrame) else None,
                "columns_num": (
                    len(df.columns) if isinstance(df, pd.DataFrame) else None
                ),
                "context": "Dataset missing values removal",
            },
        )
        raise RuntimeError(msg) from e
