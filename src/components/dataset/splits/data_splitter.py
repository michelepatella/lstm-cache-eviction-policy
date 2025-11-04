"""data_splitter.py

Utility module for splitting datasets.

This module provides the `split_dataset_data` function, which splits
a pandas DataFrame at a specified index, returning either the first
or the second portion. Useful for dividing datasets into training,
validation, and test sets.

Functions:
    split_dataset_data(
        df: pd.DataFrame,
        split_idx: int,
        take_first: bool
    ) -> pd.DataFrame
        Splits a DataFrame at a given index and returns the selected part.
"""

from typing import Any

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def split_dataset_data(
    df: pd.DataFrame,
    split_idx: int,
    take_first: bool,
) -> Any:
    """Split data at a given index.

    This function splits provided data based on the received index,
    taking the first or the second part of the data as specified.

    Args:
        df (pd.DataFrame): Data to split.
        split_idx (int): Index at which to split.
        take_first (bool): If True, take the first part of data. If False,
                           take the second part of data.

    Returns:
        Any: Split data.

    Raises:
        RuntimeError: If dataset splitting fails:
            * Invalid index type or value (TypeError, IndexError)
            * Provided object is not a DataFrame (AttributeError)
    """
    try:
        debug(
            "Dataset splitting started",
            extra={
                "split_index": split_idx,
                "take_first_part": take_first,
                "dataset_length": len(df) if hasattr(df, "__len__") else None,
                "context": "Dataset splitting",
            },
        )

        # Split data taking the first
        # or second part of it
        split_data = df[:split_idx] if take_first else df[split_idx:]

        debug(
            "Dataset splitting completed",
            extra={
                "resulting_length": (
                    len(split_data) if hasattr(split_data, "__len__") else None
                ),
                "context": "Dataset splitting",
            },
        )

        return split_data
    except (TypeError, IndexError, AttributeError) as e:
        msg = "Dataset data splitting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "split_index": split_idx,
                "take_first_part": take_first,
                "dataset_length": len(df) if hasattr(df, "__len__") else None,
                "context": "Dataset splitting",
            },
        )
        raise RuntimeError(msg) from e
