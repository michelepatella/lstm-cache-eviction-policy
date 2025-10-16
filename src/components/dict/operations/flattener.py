from typing import Any, Dict, List, Tuple
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def flatten_dict(
    nested_dict: Dict[str, Any],
    parent_key: Tuple[str, ...] = (),
) -> List[Tuple[Tuple[str, ...], Any]]:
    """
    Recursively flatten a nested dictionary.

    This function converts a nested dictionary into a flat list of tuples,
    where each tuple contains a key path (as a tuple of keys) and the
    corresponding value.

    Args:
        nested_dict (Dict[str, Any]): Dictionary to flatten.
        parent_key (Tuple[str, ...]): Accumulated key path from previous
                                      recursion levels.

    Returns:
        List[Tuple[Tuple[str, ...], Any]]: List of tuples containing the
                                           full key path and its associated value.

    Raises:
        RuntimeError: If an error occurs while flattening the dictionary, e.g.:
            * If a non-dictionary value is encountered where a dictionary is expected.
            * If recursion depth exceeds the Python limit.
            * If the input object does not have the expected attributes.
    """
    debug(f"Level to flatten dictionary at: {parent_key}")

    items: List[Tuple[Tuple[str, ...], Any]] = []

    try:
        for key, value in nested_dict.items():
            new_key = parent_key + (key,)
            if isinstance(value, dict):
                # Recurse into nested dictionary
                items.extend(flatten_dict(value, new_key))
            else:
                # Save flattened item
                items.append((new_key, value))
                debug(f"Added flattened key: {new_key} -> value: {value}")
    except (TypeError, RecursionError, AttributeError) as e:
        msg = "Failed to flatten dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    debug(f"Flattened items at level {parent_key}: {items}")
    info("Dictionary flattening completed")

    return items
