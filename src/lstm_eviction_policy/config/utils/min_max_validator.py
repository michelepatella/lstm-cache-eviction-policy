from typing import Any

from lstm_eviction_policy.utils.logs.log_utils import error


def validate_min_max(
    instance: Any, min_field: str, max_field: str, context: str
) -> Any:
    """
    Check whether a maximum value is greater than
    or equal to a minimum one.

    This function reads the least and the greatest
    values from the specified fields of the given instance,
    and checks whether the first is less than or equal to the
    second one, with respect to a given context. If the control
    is passed the function returns the instance just validated.

    Parameters:
        instance (Any): Model instance being validated.
        min_field (str): Name of the instance field representing the least value.
        max_field (str): Name of the instance field representing the greatest value.
        context (str): Context for error messages.

    Returns:
        Any: Validated model instance.

    Raises:
        ValueError: If the least value is greater than or equal to
                    the greatest one.
    """
    # Read the least and the greatest fields
    # from the given instance given the
    # corresponding fields
    min_val = getattr(instance, min_field)
    max_val = getattr(instance, max_field)

    # Check whether the minimum value is greater
    # than or equal to the maximum one
    if max_val <= min_val:
        msg = f"{context}{max_field} ({max_val}) must be greater than {context}{min_field} ({min_val})"
        error("%s", msg)
        raise ValueError(msg)

    return instance
