"""saver.py

Utility module for saving Python data structures to a YAML file.

This module provides the `save_yaml` function, which serializes a
given data dictionary into YAML format and writes it to the specified
file path, ensuring keys are sorted for consistent output.

Functions:
    save_yaml(
        data_dict: dict[str, Any],
        path: str
    ) -> None
        Writes the contents of a Python dictionary to a YAML file.
"""

from typing import Any

import yaml

from components.const import YAML_DUMP_SORT_KEYS
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_yaml(data_dict: dict[str, Any], path: str) -> None:
    """Save a data dictionary as a YAML file.

    This function saves the provided data dictionary
    as a YAML file at the specified path.

    Args:
        data_dict (dict[str, Any]): Data dictionary to save as YAML.
        path (str): Path to save YAML file at.

    Returns:
        None

    Raises:
        RuntimeError: If saving YAML file fails:
            * File cannot be written due to operating system errors (OSError).
            * File cannot be written due to permission issues (OSError).
    """
    try:
        debug(
            "YAML saving started",
            extra={
                "path": str(path),
                "keys_num": (
                    len(data_dict) if isinstance(data_dict, dict) else None
                ),
                "context": "YAML saving",
            },
        )

        # Save provided data dictionary at
        # specified path
        with open(path, "w") as f:
            yaml.dump(data_dict, f, sort_keys=YAML_DUMP_SORT_KEYS)

        debug(
            "YAML saving completed",
            extra={
                "path": str(path),
                "keys_num": (
                    len(data_dict) if isinstance(data_dict, dict) else None
                ),
                "context": "YAML saving",
            },
        )
    except OSError as e:
        msg = "YAML saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "keys_num": (
                    len(data_dict) if isinstance(data_dict, dict) else None
                ),
                "context": "YAML saving",
            },
        )
        raise RuntimeError(msg) from e
