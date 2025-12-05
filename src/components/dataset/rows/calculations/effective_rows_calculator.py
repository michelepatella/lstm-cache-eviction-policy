"""effective_rows_calculator.py

Utility module for computing the effective number of rows in a dataset.

This module provides the `calculate_effective_dataset_rows` function, which
calculates the number of usable rows in a DataFrame with respect to a
specified sequence length. It ensures that the resulting number of rows
is non-negative and suitable for sequence extraction.

Functions:
    calculate_effective_dataset_rows(
        df: pd.DataFrame,
        seq_len: int
    ) -> int
        Returns the effective number of dataset rows given a sequence length.
"""

import pandas as pd

from components.logs.levels.error_logger import error


def calculate_effective_dataset_rows(df: pd.DataFrame, seq_len: int) -> int:
    """Calculate effective dataset rows.

    This function calculates and returns the effective dataset
    rows, with respect to the sequence length provided.

    Args:
        df (pd.DataFrame): Dataset to calculate effective length for.
        seq_len (int): Length of sequences to extract.

    Returns:
        int: Number of available sequences.

    Raises:
        RuntimeError: If dataset effective rows calculation fails:
            * Invalid arguments or data types (TypeError, ValueError).
            * Dataset is not a DataFrame (AttributeError).
    """
    try:
        # Calculate dataset length
        dataset_length = len(df) - seq_len

        # Check whether dataset length is negative
        if dataset_length < 0:
            msg = "Calculated effective dataset length is negative"
            error(
                msg,
                extra={
                    "num_rows": len(df) if hasattr(df, "__len__") else None,
                    "seq_len": seq_len,
                    "context": "Effective dataset length calculation",
                },
            )
            raise RuntimeError(msg)

        return dataset_length
    except (TypeError, AttributeError, ValueError) as e:
        msg = "Effective dataset length calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "num_rows": len(df) if hasattr(df, "__len__") else None,
                "seq_len": seq_len,
                "context": "Effective dataset length calculation",
            },
        )
        raise RuntimeError(msg) from e
