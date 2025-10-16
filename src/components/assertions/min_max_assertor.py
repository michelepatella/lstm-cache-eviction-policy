from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def assert_min_less_than_max(
    min_val: int | float,
    max_val: int | float,
    context: str,
) -> None:
    """
    Check whether a maximum value is greater than
    or equal to a minimum one.

    This function, given the least and the greatest
    values, checks whether the first is less than or equal to the
    second one, with respect to a given context.

    Args:
        min_val (int | float): The least value.
        max_val (int | float): The greatest value.
        context (str): Context for error messages.

    Returns:
        None

    Raises:
        ValueError: If the minimum value is greater
                    than or equal to the maximum.
        RuntimeError: If an error occurs during validation.
    """
    debug(
        f"Min/Max values to be validated: {min_val}, {max_val} from {context}"
    )

    try:
        # Check whether the minimum value is greater
        # than or equal to the maximum one
        if max_val <= min_val:
            msg = f"{max_val} must be greater than {min_val} ({context})"
            error("%s", msg)
            raise ValueError(msg)
    except TypeError as e:
        msg = "Failed to compare min/max values"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"{min_val} and {max_val} validated for {context}")
