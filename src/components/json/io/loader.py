import json
from typing import Any, Dict, Optional, Union

from box import Box

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import JSON_WRAP_BOX_DEFAULT


def load_json(
    path: str, wrap: Optional[bool] = JSON_WRAP_BOX_DEFAULT
) -> Union[Dict[Any, Any], Box]:
    """
    Load a JSON file.

    This function loads a JSON file from the specified path.
    Optionally, it wraps the resulting dictionary in a Box object
    to allow dot-notation access.

    Args:
        path (str): Path to load JSON file from.
        wrap (Optional[bool]): Whether to wrap the output in Box.

    Returns:
        Union[Dict[Any, Any], Box]: Loaded JSON data, optionally wrapped in Box.

    Raises:
        RuntimeError: If loading the JSON file fails:
            * Opening or reading the file fails due to missing file, permission issues,
              or other I/O errors (OSError).
            * Decoding the JSON content fails because the file is not valid JSON
              (json.JSONDecodeError).
    """
    try:
        debug(f"Path to load JSON from: {path}")

        # Load JSON data from file at
        # specified path
        with open(path, "r") as f:
            json_data = json.load(f)

        info(f"JSON loaded from: {path}")

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
