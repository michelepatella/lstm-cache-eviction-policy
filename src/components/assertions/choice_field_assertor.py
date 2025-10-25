from typing import Any, List

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def assert_choice_field(
    field_value: Any,
    allowed_field_values: List[Any],
    field_context: str,
) -> None:
    """
    Check whether a choice field is valid or not.

    This function validates the choice field, ensuring its value it's
    a valid one, according to a list of allowed values.

    Args:
        field_value (Any): Value of the choice field.
        allowed_field_values (List[Any]): List of allowed values.
        field_context (str): Context for error messages.

    Returns:
        None

    Raises:
        ValueError: If the choice field is not valid.
        TypeError: If allowed field values data structure is not a list.
    """
    debug(
        "Choice field validation started",
        extra={
            "field_value": field_value,
            "allowed_field_values": allowed_field_values,
            "field_context": field_context,
            "context": "Choice field assertion",
        },
    )

    # Check whether allowed field values
    # is not a list
    if not isinstance(allowed_field_values, list):
        msg = "Allowed field values must be a list"
        error(
            msg,
            extra={
                "field_value": field_value,
                "field_value_type": type(field_value).__name__,
                "allowed_field_values": allowed_field_values,
                "allowed_field_values_type": type(
                    allowed_field_values
                ).__name__,
                "field_context": field_context,
                "status": "Invalid",
                "context": "Choice field assertion",
            },
        )
        raise TypeError(msg)

    # Check whether the field has not
    # allowed value
    if field_value not in allowed_field_values:
        msg = "Invalid field value"
        error(
            msg,
            extra={
                "field_value": field_value,
                "field_value_type": type(field_value).__name__,
                "allowed_field_values": allowed_field_values,
                "allowed_field_values_type": type(
                    allowed_field_values
                ).__name__,
                "field_context": field_context,
                "status": "Invalid",
                "context": "Choice field assertion",
            },
        )
        raise ValueError(msg)

    debug(
        "Choice field validation completed",
        extra={
            "field_value": field_value,
            "allowed_field_values": allowed_field_values,
            "field_context": field_context,
            "status": "Valid",
            "context": "Choice field assertion",
        },
    )
