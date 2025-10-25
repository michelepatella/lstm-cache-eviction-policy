from typing import Any, Dict, List, Tuple

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
        RuntimeError: If flattening the dictionary fails:
            * Input is not a dictionary (TypeError).
            * Maximum recursion depth exceeded due to overly nested dictionaries
              (RecursionError).
            * Nested dictionary contains invalid structure that prevents iteration
              (AttributeError).
    """
    try:
        debug(
            "Dictionary flattering started",
            extra={
                "parent_key": parent_key,
                "nested_dict_type": type(nested_dict).__name__,
                "num_keys_at_level": (
                    len(nested_dict) if isinstance(nested_dict, dict) else None
                ),
                "context": "Dictionary flattening",
            },
        )

        items = []
        for key, value in nested_dict.items():
            new_key = parent_key + (key,)
            if isinstance(value, dict):
                # Recurse into nested dictionary
                items.extend(flatten_dict(value, new_key))
            else:
                # Save flattened item
                items.append((new_key, value))
                debug(
                    "Flattened key added",
                    extra={
                        "flattened_key": new_key,
                        "value": value,
                        "context": "Dictionary flattening",
                    },
                )

        debug(
            "Dictionary flattering completed",
            extra={
                "parent_key": parent_key,
                "num_flattened_items": len(items),
                "context": "Dictionary flattening",
            },
        )

        return items
    except (TypeError, RecursionError, AttributeError) as e:
        msg = "Dictionary flattering failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "parent_key": parent_key,
                "nested_dict_type": type(nested_dict).__name__,
                "context": "Dictionary flattening",
            },
        )
        raise RuntimeError(msg) from e
