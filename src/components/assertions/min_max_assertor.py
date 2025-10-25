from typing import Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def assert_min_less_than_max(
    min_val: Union[int, float],
    max_val: Union[int, float],
    values_context: str,
) -> None:
    """
    Check whether a maximum value is greater than or equal to
    a minimum one or not.

    This function, given the minimum and the maximum values, checks
    whether the first is less than or equal to the second one.

    Args:
        min_val (Union[int, float]): The maximum value.
        max_val (Union[int, float]): The minimum value.
        values_context (str): Context for error messages.

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
            "Min/max validation started",
            extra={
                "min_val": min_val,
                "max_val": max_val,
                "values_context": values_context,
                "context": "Min/max assertion",
            },
        )

        # Check whether the minimum value is greater
        # than or equal to the maximum one
        if max_val <= min_val:
            msg = "Invalid min/max values"
            error(
                msg,
                extra={
                    "min_val": min_val,
                    "max_val": max_val,
                    "min_val_type": type(min_val).__name__,
                    "max_val_type": type(max_val).__name__,
                    "values_context": values_context,
                    "status": "Invalid",
                    "context": "Min/max assertion",
                },
            )
            raise ValueError(msg)

        debug(
            "Min/max validation completed",
            extra={
                "min_val": min_val,
                "max_val": max_val,
                "values_context": values_context,
                "status": "Valid",
                "context": "Min/max assertion",
            },
        )
    except TypeError as e:
        msg = "Min/max validation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "min_val": min_val,
                "max_val": max_val,
                "min_val_type": type(min_val).__name__,
                "max_val_type": type(max_val).__name__,
                "values_context": values_context,
                "status": "Invalid",
                "context": "Min/max assertion",
            },
        )
        raise RuntimeError(msg) from e
