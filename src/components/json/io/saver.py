"""saver.py

Utility module for saving dictionaries or lists of dictionaries to JSON files.

This module provides the `save_json` function, which serializes Python data
(dictionaries or lists of dictionaries) into a JSON file on disk with a
configurable indentation level.

Functions:
    save_json(
        data_dict: dict | list[dict],
        path: str,
        json_indent: int = JSON_INDENT
    ) -> None
        Save a Python dictionary or a list of dictionaries as a JSON file
        at the specified path with optional indentation.
"""

import json

from components.const import JSON_INDENT
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_json(
    data_dict: dict | list[dict],
    path: str,
    json_indent: int = JSON_INDENT,
) -> None:
    """Save a data dictionary as a JSON file.

    This function saves a data dictionary as a
    JSON file at the specified path.

    Args:
        data_dict (dict | list[dict]): Data to save.
        path (str): File path where the JSON will be saved.
        json_indent (int): Indent for the JSON file.

    Returns:
        None

    Raises:
        RuntimeError: If saving the JSON file fails:
            * Serialization of the data dictionary fails because one or more
              values cannot be encoded into valid JSON (TypeError).
            * Writing to the file fails due to missing path, permission issues,
              or other I/O errors (OSError).
    """
    try:
        debug(
            "JSON saving started",
            extra={
                "path": str(path),
                "json_indent": json_indent,
                "data_type": type(data_dict).__name__,
                "items_num": (
                    len(data_dict)
                    if isinstance(data_dict, (dict, list))
                    else None
                ),
                "context": "JSON saving",
            },
        )

        # Save data dictionary as JSON file
        # to the specified path
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=json_indent)

        debug(
            "JSON saving completed",
            extra={
                "path": str(path),
                "json_indent": json_indent,
                "data_type": type(data_dict).__name__,
                "items_num": (
                    len(data_dict)
                    if isinstance(data_dict, (dict, list))
                    else None
                ),
                "context": "JSON saving",
            },
        )
    except (TypeError, OSError) as e:
        msg = "JSON saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "json_indent": json_indent,
                "data_type": type(data_dict).__name__,
                "context": "JSON saving",
            },
        )
        raise RuntimeError(msg) from e
