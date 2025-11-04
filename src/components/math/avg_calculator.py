"""avg_calculator.py

Utility module for calculating averages.

This module provides the `calculate_average` function, which
computes the mean of a list of numeric values, returning None
if the list is empty.

Functions:
    calculate_average(values: list[int | float]) -> float | None
        Computes the arithmetic mean of the provided list of numbers.
"""

from components.logs.levels.error_logger import error


def calculate_average(values: list[int | float]) -> float | None:
    """Calculate the average of a list of numeric values.

    This function computes the average of a list of numeric values.

    Args:
        values (list[int | float]): List of numeric values to average.

    Returns:
        float | None: Average value (None if no values are provided).

    Raises:
        RuntimeError: If average calculation fails:
            * List contains non-numeric values (TypeError).
    """
    try:
        # Calculate average of values if there
        # is at least one element in the list
        avg_value = None
        if len(values) != 0:
            avg_value = sum(values) / len(values)

        return avg_value
    except TypeError as e:
        msg = "Average calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "values_type": type(values).__name__,
                "values_num": len(values) if values is not None else 0,
                "context": "Average calculation",
            },
        )
        raise RuntimeError(msg) from e
