from typing import Any, Dict, Sequence

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


def set_nested_dict(
    nested_dict: Dict[str, Any],
    keys: Sequence[str],
    value: int | float,
) -> None:
    """
    Set a value in a nested dictionary given a sequence of keys.

    This function traverses a nested dictionary according
    to the provided sequence of keys, creating intermediate
    dictionaries if needed, and sets the value at the last key.

    Parameters:
        nested_dict (Dict[str, Any]): Dictionary to update.
        keys (Sequence[str]): Sequence of nested keys leading to the value.
        value (int | float): Value to set at the nested location.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while setting a nested dictionary value, e.g.:
            * If nested dictionary is not a dictionary or keys is not iterable.
            * If keys sequence is empty.
            * If assignment fails due to an invalid key.
    """
    try:
        # Initialization
        current_dict = nested_dict

        # Iterate over all the keys except
        # the last one to go down the nested levels
        for key in keys[:-1]:
            # If the current key is not in the
            # current dictionary or this latter is not
            # a dictionary, make it one
            if key not in current_dict or not isinstance(
                current_dict[key], dict
            ):
                current_dict[key] = {}

            # Go down a level more
            current_dict = current_dict[key]

        # Set the desired value in the last
        # position
        current_dict[keys[-1]] = value

        debug(f"Set value at nested key path: {keys} -> {value}")
    except (TypeError, IndexError, KeyError) as e:
        msg = "Failed to set value in nested dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Nested dictionary setting completed")
