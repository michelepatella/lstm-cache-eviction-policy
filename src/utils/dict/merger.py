from typing import Dict, Any

from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def merge_dicts(
    original: Dict[str, Any] | None,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    This function, given two dictionaries, merges updated
    values of the updated dictionary into the original one,
    returning the resulting dictionary.

    Args:
        original (Dict[str, Any] | None): Original dictionary.
        updates (dict): Dictionary containing updates to apply.

    Returns:
        Dict[str, Any]: Merged dictionary.
    """
    # Check whether the original dictionary
    # is None to set it as empty
    if original is None:
        original = {}

    try:
        # Merge dictionaries recursively
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(original.get(key), dict):
                # If the value is still a dictionary,
                # apply merge recursively
                original[key] = merge_dicts(original[key], value)
            else:
                # Set the final value
                original[key] = value
    except TypeError as e:
        msg = "Failed to merge dictionaries"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Dictionaries merged")

    return original
