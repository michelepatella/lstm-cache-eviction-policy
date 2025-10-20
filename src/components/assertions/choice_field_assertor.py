from typing import Any, List

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def assert_choice_field(
    field_value: Any,
    allowed_field_values: List[Any],
    context: str,
) -> None:
    """
    Check whether a choice field is valid or not.

    This function validates the choice field, ensuring its value it's
    a valid one, according to a list of allowed values.

    Args:
        field_value (Any): Value of the choice field.
        allowed_field_values (List[Any]): List of allowed values.
        context (str): Context for error messages.

    Returns:
        None

    Raises:
        ValueError: If the choice field is not valid.
        TypeError: If allowed field values data structure is not a list.
    """
    debug(f"Field value to be validated: {field_value}:")
    debug(f"Allowed field values: {allowed_field_values}:")

    # Check whether allowed field values
    # is not a list
    if not isinstance(allowed_field_values, list):
        msg = f"Allowed field values the must be a list ({context})"
        error("%s", msg)
        raise TypeError(msg)

    # Check whether the field has not
    # allowed value
    if field_value not in allowed_field_values:
        msg = (
            f"{context} must be one of the following"
            f" values: {allowed_field_values}"
        )
        error("%s", msg)
        raise ValueError(msg)

    debug(f"{context}: {field_value}, validated")
