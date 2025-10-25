from typing import Literal

import pandas as pd

from components.const import (
    MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    MISSING_VALUES_REMOVAL_DROPNA_HOW,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
        debug(
            "Dataset missing values removal started",
            extra={
                "num_rows_before": (
                    len(df) if isinstance(df, pd.DataFrame) else None
                ),
                "num_columns_before": (
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
                "num_rows_after": len(new_df),
                "num_columns_after": len(new_df.columns),
                "num_removed_rows": (
                    len(df) - len(new_df) if dropna_axis == 0 else None
                ),
                "num_removed_columns": (
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
                "num_rows": len(df) if isinstance(df, pd.DataFrame) else None,
                "num_columns": (
                    len(df.columns) if isinstance(df, pd.DataFrame) else None
                ),
                "context": "Dataset missing values removal",
            },
        )
        raise RuntimeError(msg) from e
