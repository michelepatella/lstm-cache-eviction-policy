from typing import Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def assert_min_less_than_max(
    min_val: Union[int, float],
    max_val: Union[int, float],
    context: str,
) -> None:
    """
    Check whether a maximum value is greater than or equal to
    a minimum one or not.

    This function, given the minimum and the maximum values, checks
    whether the first is less than or equal to the second one.

    Args:
        min_val (Union[int, float]): The maximum value.
        max_val (Union[int, float]): The minimum value.
        context (str): Context for error messages.

    Returns:
        None

    Raises:
        ValueError: If the minimum value is greater than or equal to
                    the maximum one.
        RuntimeError: If validation fails due to invalid types preventing
                      comparison (TypeError).
    """
    try:
        debug(
            f"Min/Max values to be validated: {min_val}, {max_val} from {context}"
        )

        # Check whether the minimum value is greater
        # than or equal to the maximum one
        if max_val <= min_val:
            msg = f"{max_val} must be greater than {min_val} ({context})"
            error("%s", msg)
            raise ValueError(msg)

        info(f"{min_val} and {max_val} validated for {context}")
    except TypeError as e:
        msg = "Failed to validate minimum and maximum values"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
