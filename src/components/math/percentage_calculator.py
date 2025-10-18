from typing import Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_percentage(
    value: Union[int, float], total: Union[int, float]
) -> float:
    """
    Calculate the percentage of a value.

    This function, given a total, calculates the percentage of
    a provided value.

    Args:
        value (Union[int, float]): Value for which to calculate the percentage.
        total (Union[int, float]): The total used for calculating the percentage.

    Returns:
        float: Percentage of the value.

    Raises:
        RuntimeError: If the value and/or total are not valid inputs.
    """
    try:
        debug(f"Value: {value}, and total: {total} to calculate percentage")

        # If total is not zero calculate
        # percentage, otherwise set
        # percentage to zero
        if total == 0:
            percentage = 0.0
        else:
            percentage = value / total * 100

        info(f"Percentage calculated: {percentage}%")

        return percentage
    except (TypeError, ZeroDivisionError) as e:
        msg = "Failed to calculate percentage"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
