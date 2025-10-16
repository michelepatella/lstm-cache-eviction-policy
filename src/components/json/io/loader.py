import json
from typing import Any, Dict

from box import Box

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import JSON_WRAP_BOX_DEFAULT


def load_json(
    path: str, wrap: bool = JSON_WRAP_BOX_DEFAULT
) -> Dict[Any, Any] | Box:
    """
    Load a JSON file.

    This function loads a JSON file from the specified path.
    Optionally, it wraps the resulting dictionary in a Box object
    to allow dot-notation access.

    Args:
        path (str): Path to the JSON file.
        wrap (bool): Whether to wrap the output in Box.

    Returns:
        Dict[Any, Any] | Box: Loaded JSON data, optionally wrapped in Box.

    Raises:
        RuntimeError: If an error occurs while loading the JSON file, e.g.:
            * File not found or inaccessible.
            * JSON decoding error.
            * I/O error while reading.
    """
    debug(f"Path where to load JSON: {path}")

    try:
        # Load JSON data from file at
        # specified path
        with open(path, "r") as f:
            json_data = json.load(f)

        info(f"JSON loaded from {path}")

        # Check whether to wrap JSON
        # data into a Box object, allowing
        # dot-notation access
        if wrap:
            return Box(json_data)
        else:
            return json_data
    except (OSError, json.JSONDecodeError) as e:
        msg = "Failed to load JSON"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
