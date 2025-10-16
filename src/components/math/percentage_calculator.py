from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_percentage(counter: int | float, total: int | float) -> float:
    """
    Calculate the percentage of a counter.

    This function, given a total, calculates the percentage of
    a provided counter.

    Args:
        counter (int | float): Counter for which to calculate the percentage.
        total (int | float): The total used for calculating the percentage.

    Returns:
        float: Percentage of the counter.

    Raises:
        RuntimeError: If the counter and/or total are not valid inputs.
    """
    debug(f"Counter: {counter}, and total: {total}, to calculate % for")

    # Check inputs validity
    if counter < 0 or total < 0:
        msg = (
            "Counter and total must be non-negative numbers for % calculation"
        )
        error("%s", msg)
        raise RuntimeError(msg)

    if not isinstance(counter, int | float) or not isinstance(
        total, int | float
    ):
        msg = "Counter and total must be integers or floats for % calculation"
        error("%s", msg)
        raise RuntimeError(msg)

    if counter > total:
        msg = "Counter cannot be greater than total for % calculation"
        error("%s", msg)
        raise RuntimeError(msg)

    # Calculate percentage
    if total == 0:
        # If total is zero, return zero
        percentage = 0.0
    else:
        percentage = counter / total * 100

    info(f"Percentage calculated: {percentage}%")

    return percentage
