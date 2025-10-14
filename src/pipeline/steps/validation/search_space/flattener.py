from typing import Any, Dict, List, Tuple

from pipeline.const import VALIDATION_PARAMS_SUFFIX
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def flatten_search_space(
    nested_dict: Dict[str, Any], parent_key: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], Any]]:
    """
    Recursively flatten a nested search space dictionary.

    This function converts a nested dictionary representing the search space
    into a flat list of tuples, where each tuple contains a key path and
    the corresponding list of possible values.

    Args:
        nested_dict (Dict[str, Any]): The search space dictionary to flatten.
        parent_key (Tuple[str, ...]): Accumulated key path from
                                      previous recursion levels.

    Returns:
        List[Tuple[Tuple[str, ...], Any]]: List of tuples containing
                                           the key path and its associated
                                           list of values.

    Raises:
        RuntimeError: If an error occurs while
                      flattening the search space, e.g.:
            * If a non-iterable value is
              encountered where a dict is expected.
            * If recursion depth exceeds the Python limit.
            * If the input dictionary does not
              have expected attributes.
    """
    debug(f"Flattening search space at level: {parent_key}")

    # Initialize list of items
    items: List[Tuple[Tuple[str, ...], Any]] = []

    try:
        for k, v in nested_dict.items():
            # Clean parameter name by removing suffix
            clean_key = k.replace(VALIDATION_PARAMS_SUFFIX, "")
            new_key = parent_key + (clean_key,)

            if isinstance(v, dict):
                # Recurse into nested dictionary
                items.extend(flatten_search_space(v, new_key))
            else:
                # Save item
                items.append((new_key, v))

                debug(f"Added flattened key: {new_key} -> values: {v}")
    except (TypeError, RecursionError, AttributeError) as e:
        msg = "Failed to flatten search space"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    debug(f"Flattened items at level {parent_key}: {items}")

    info("Search space flattening completed")

    return items
