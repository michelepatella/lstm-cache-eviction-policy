from typing import Any, Dict, Union

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
                original_dict.get(key), dict
            ):
                debug(
                    "Merging nested dictionary for key",
                    extra={
                        "nested_key": key,
                        "context": "Dictionaries merging",
                    },
                )
                # If the value is still a dictionary,
                # apply merge recursively
                original_dict[key] = merge_dicts(original_dict[key], value)
            else:
                # Set the final value
                original_dict[key] = value
                debug(
                    "Key to value merged",
                    extra={
                        "key": key,
                        "value": value,
                        "context": "Dictionaries merging",
                    },
                )

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
