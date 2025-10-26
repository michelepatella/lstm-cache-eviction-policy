from collections.abc import Sequence
from typing import Any

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def set_dict_value(
    data_dict: dict[str, Any],
    keys: Sequence[str],
    value: Any,
) -> None:
    """Set a value in a (nested) dictionary given a sequence of keys.

    This function traverses a (nested) dictionary according to the
    provided sequence of one or more keys, setting the value at the
    last key.

    Args:
        data_dict (dict[str, Any]): Dictionary to update.
        keys (Sequence[str]): Sequence of one or more keys leading
                              to the value.
        value (Any): Value to set.

    Returns:
        None

    Raises:
        RuntimeError: If setting dictionary value fails:
            * Invalid types for data dictionary or keys (TypeError).
            * Non-iterable keys (AttributeError).
    """
    try:
        debug(
            "Dictionary value setting started",
            extra={
                "keys_to_set": keys,
                "value_to_set": value,
                "context": "Dictionary value setting",
            },
        )

        # Iterate over all the keys except
        # the last one to go down the nested levels
        current_dict = data_dict
        for key in keys[:-1]:
            # If the current key is not in the
            # current dictionary or this latter is not
            # a dictionary, make it one
            if key not in current_dict or not isinstance(
                current_dict[key],
                data_dict,
            ):
                current_dict[key] = {}

            # Go down a level more
            current_dict = current_dict[key]

        # Set the desired value in the last
        # position
        current_dict[keys[-1]] = value

        debug(
            "Dictionary value setting completed",
            extra={
                "keys_final": keys,
                "value_final": value,
                "context": "Dictionary value setting",
            },
        )
    except (TypeError, AttributeError) as e:
        msg = "Dictionary value setting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "keys_attempted": keys,
                "value_attempted": value,
                "context": "Dictionary value setting",
            },
        )
        raise RuntimeError(msg) from e
