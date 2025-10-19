from typing import Any, Dict, Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def merge_dicts(
    original_dict: Union[Dict[str, Any], None],
    updated_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    This function, given two dictionaries, merges updated values of
    the updated dictionary into the original one, returning the
    resulting dictionary.

    Args:
        original_dict (Union[Dict[str, Any], None]): Original dictionary.
        updated_dict (dict): Dictionary containing updates to apply.

    Returns:
        Dict[str, Any]: Merged dictionary.

    Raises:
        RuntimeError: If merging dictionaries fails:
            * One or both dictionaries are of incorrect type (TypeError).
    """
    try:
        # Check whether the original dictionary
        # is None to initialize it
        if original_dict is None:
            original_dict = {}

        # Merge dictionaries recursively
        for key, value in updated_dict.items():
            if isinstance(value, dict) and isinstance(
                original_dict.get(key), dict
            ):
                debug(f"Merging nested dictionary for key '{key}'")
                # If the value is still a dictionary,
                # apply merge recursively
                original_dict[key] = merge_dicts(original_dict[key], value)
            else:
                # Set the final value
                original_dict[key] = value
                debug(f"Merged key '{key}' in dictionaries to value '{value}'")

        info("Dictionaries merged")

        return original_dict
    except TypeError as e:
        msg = "Failed to merge dictionaries"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
