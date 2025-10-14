from typing import Any, Dict, Sequence

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def set_dict_value(
    data: Dict[str, Any],
    keys: Sequence[str],
    value: int | float,
) -> None:
    """
    Set a value in a (nested) dictionary given a sequence 
    of one or more keys.

    This function traverses a (nested) dictionary according
    to the provided sequence of one or more keys, setting the
    value at the last key.

    Args:
        data (Dict[str, Any]): Dictionary to update.
        keys (Sequence[str]): Sequence of one or more keys leading to the value.
        value (int | float): Value to set.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while setting
                      a (nested) dictionary value, e.g.:
            * If data is not a dictionary or key(s) is not iterable.
            * If key(s) sequence is empty.
            * If assignment fails due to an invalid key.
    """
    try:
        # Initialization
        current_dict = data

        # Iterate over all the keys except
        # the last one to go down the nested levels
        for key in keys[:-1]:
            # If the current key is not in the
            # current dictionary or this latter is not
            # a dictionary, make it one
            if key not in current_dict or not isinstance(
                current_dict[key], data
            ):
                current_dict[key] = {}

            # Go down a level more
            current_dict = current_dict[key]

        # Set the desired value in the last
        # position
        current_dict[keys[-1]] = value

        debug(f"Set dictionary value: {keys} -> {value}")
    except (TypeError, IndexError, KeyError) as e:
        msg = "Failed to set value in dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Value set in dictionary")
