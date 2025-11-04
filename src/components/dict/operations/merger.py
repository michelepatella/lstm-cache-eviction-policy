"""merger.py

Utility module for recursively merging dictionaries.

This module provides the `merge_dicts` function, which merges two dictionaries
by updating the original dictionary with values from the updated dictionary.
Nested dictionaries are merged recursively, allowing deep updates without
overwriting entire sub-dictionaries.

Functions:
    merge_dicts(
        original_dict: dict[str, Any] | None,
        updated_dict: dict[str, Any]
    ) -> dict[str, Any]
        Recursively merge two dictionaries, returning the merged result.
"""

from typing import Any

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def merge_dicts(
    original_dict: dict[str, Any] | None,
    updated_dict: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    This function, given two dictionaries, merges updated values of
    the updated dictionary into the original one, returning the
    resulting dictionary.

    Args:
        original_dict (dict[str, Any] | None): Original dictionary.
        updated_dict (dict[str, Any]): Dictionary containing updates to apply.

    Returns:
        dict[str, Any]: Merged dictionary.

    Raises:
        RuntimeError: If merging dictionaries fails:
            * One or both dictionaries are of incorrect type (TypeError).
    """
    try:
        # Check whether the original dictionary
        # is None to initialize it
        if original_dict is None:
            original_dict = {}

        debug(
            "Dictionaries merging started",
            extra={
                "original_dict_keys": list(original_dict.keys()),
                "updated_dict_keys": list(updated_dict.keys()),
                "context": "Dictionaries merging",
            },
        )

        # Merge dictionaries recursively
        for key, value in updated_dict.items():
            if isinstance(value, dict) and isinstance(
                original_dict.get(key),
                dict,
            ):
                # If the value is still a dictionary,
                # apply merge recursively
                original_dict[key] = merge_dicts(original_dict[key], value)
            else:
                # Set the final value
                original_dict[key] = value

        debug(
            "Dictionaries merging completed",
            extra={
                "merged_keys": list(original_dict.keys()),
                "context": "Dictionaries merging",
            },
        )

        return original_dict
    except TypeError as e:
        msg = "Dictionaries merging failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "original_dict_type": type(original_dict).__name__,
                "updated_dict_type": type(updated_dict).__name__,
                "context": "Dictionaries merging",
            },
        )
        raise RuntimeError(msg) from e
