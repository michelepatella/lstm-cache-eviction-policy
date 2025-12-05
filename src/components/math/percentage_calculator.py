"""percentage_calculator.py

Utility module for computing percentages.

This module provides the `calculate_percentage` function, which computes
the percentage of a given value relative to a total.

Functions:
    calculate_percentage(
        value: float,
        total: float
    ) -> float
        Returns the percentage representation of value with respect to total.
"""

from components.logs.levels.error_logger import error


def calculate_percentage(
    value: float,
    total: float,
) -> float:
    """Calculate the percentage of a value.

    This function, given a total, calculates the percentage of
    a provided value.

    Args:
        value (float): Value for which to calculate
                       the percentage.
        total (float): The total used for calculating
                       the percentage.

    Returns:
        float: Percentage of the value.

    Raises:
        RuntimeError: If the value and/or total are not valid inputs.
    """
    try:
        # If total is not zero calculate
        # percentage, otherwise set
        # percentage to zero
        if total == 0:
            percentage = 0.0
        else:
            percentage = value / total * 100

        return percentage
    except (TypeError, ZeroDivisionError) as e:
        msg = "Percentage calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "current_value": value,
                "current_value_type": type(value).__name__,
                "total": total,
                "total_type": type(total).__name__,
                "context": "Percentage calculation",
            },
        )
        raise RuntimeError(msg) from e
