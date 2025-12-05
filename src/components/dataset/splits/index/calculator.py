"""calculator.py

Utility module for computing split indices in datasets.

This module provides the `calculate_dataset_split_index` function, which
calculates the split index of a dataset based on its length and a given
fraction.

Functions:
    calculate_dataset_split_index(
        dataset_len: int,
        split_perc: float
    ) -> int
        Computes the integer index at which to split a dataset according
        to a specified percentage.
"""

from components.logs.levels.error_logger import error


def calculate_dataset_split_index(dataset_len: int, split_perc: float) -> int:
    """Calculate split index for a dataset.

    This function calculates the split index for a given dataset,
    according to the provided percentage.

    Args:
        dataset_len (int): Length of the dataset.
        split_perc (float): Fraction to split.

    Returns:
        int: Dataset split index.

    Raises:
        RuntimeError: If split index calculation fails:
            * Invalid dataset length or split percentage type (TypeError).
            * Invalid numeric value (ValueError).
    """
    try:
        # Calculate split index
        return int(dataset_len * split_perc)
    except (TypeError, ValueError) as e:
        msg = "Dataset split index calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dataset_length": dataset_len,
                "split_percentage": split_perc,
                "context": "Dataset split index calculation",
            },
        )
        raise RuntimeError(msg) from e
