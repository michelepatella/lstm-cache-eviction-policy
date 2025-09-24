from typing import Any, List

from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def validate_choice_field(
    instance: Any,
    field_value: Any,
    allowed_field_values: List[Any],
    context: str,
) -> Any:
    """
    Check whether a choice field
    is valid or not.

    This function validates the choice field,
    ensuring its value it's a valid one, according
    to a list of allowed values.

    Parameters:
        instance (Any): Model instance being validated.
        field_value (Any): Value of the choice field.
        allowed_field_values (List[Any]): List of allowed values.
        context (str): Context for error messages.

    Returns:
        Any: Validated model instance.

    Raises:
        ValueError: If the choice field is not valid (i.e.,
                    is not a value among allowed ones).
    """
    debug(f"Field value to be validated: {field_value}:")
    debug(f"Allowed field values: {allowed_field_values}:")

    # Check whether the field value has not
    # allowed type
    if field_value not in allowed_field_values:
        msg = f"{context} must be one of the following values: {allowed_field_values}"
        error("%s", msg)
        raise ValueError(msg)

    info(f"{context}: {field_value}, validated")

    return instance
