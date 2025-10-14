from typing import List
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def calculate_average(values: List[int | float]) -> float | None:
    """
    Calculate the average of a list of numeric values.

    This function computes the average of a list
    of numeric values (integers or floats). If the list
    is empty, the function returns None.

    Args:
        values (List[int | float]): List of numeric values to average.

    Returns:
        float | None: Average value, or None if
                      no values are provided.
    """
    avg_value = None

    debug(f"Number of values to average: {len(values)}")

    # If there is at least a value
    # in the list
    if len(values) != 0:
        # Calculate average value
        avg_value = sum(values) / len(values)

    info(f"Average value calculated: {avg_value}")

    return avg_value