from typing import List, Optional, Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_average(values: List[Union[int, float]]) -> Optional[float]:
    """
    Calculate the average of a list of numeric values.

    This function computes the average of a list of numeric values.

    Args:
        values (List[Union[int, float]]): List of numeric values to average.

    Returns:
        Optional[float]: Average value (None if no values are provided).

    Raises:
        RuntimeError: If average calculation fails:
            * List contains non-numeric values (TypeError).
    """
    try:
        debug(f"Number of values to average: {len(values)}")

        # Calculate average of values if there
        # is at least one element in the list
        avg_value = None
        if len(values) != 0:
            avg_value = sum(values) / len(values)

        debug(f"Average value calculated: {avg_value}")

        return avg_value
    except TypeError as e:
        msg = "Failed to calculate average"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
