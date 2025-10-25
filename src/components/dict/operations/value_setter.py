from typing import Any, Dict, Sequence

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def set_dict_value(
    data_dict: Dict[str, Any],
    keys: Sequence[str],
    value: Any,
) -> None:
    """
    Set a value in a (nested) dictionary given a sequence of keys.

    This function traverses a (nested) dictionary according to the
    provided sequence of one or more keys, setting the value at the
    last key.

    Args:
        data_dict (Dict[str, Any]): Dictionary to update.
        keys (Sequence[str]): Sequence of one or more keys leading to the value.
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
        for keys in keys[:-1]:
            # If the current key is not in the
            # current dictionary or this latter is not
            # a dictionary, make it one
            if keys not in current_dict or not isinstance(
                current_dict[keys], data_dict
            ):
                current_dict[keys] = {}

            # Go down a level more
            current_dict = current_dict[keys]

        # Set the desired value in the last
        # position
        current_dict[keys[-1]] = value

        debug(
            "Dictionary value setting completed",
            extra={
                "final_keys": keys,
                "final_value": value,
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
