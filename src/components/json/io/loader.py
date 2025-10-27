import json
from typing import Any

from box import Box

from components.const import JSON_WRAP_BOX_ENABLED
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def load_json(
    path: str,
    wrap: bool = JSON_WRAP_BOX_ENABLED,
) -> dict[Any, Any] | Box:
    """Load a JSON file.

    This function loads a JSON file from the specified path.
    Optionally, it wraps the resulting dictionary in a Box object
    to allow dot-notation access.

    Args:
        path (str): Path to load JSON file from.
        wrap (bool): Whether to wrap the output in Box.

    Returns:
        dict[Any, Any] | Box: Loaded JSON data, optionally
                              wrapped in Box.

    Raises:
        RuntimeError: If loading the JSON file fails:
            * Opening or reading the file fails due to missing file,
              permission issues, or other I/O errors (OSError).
            * Decoding the JSON content fails because the file is not
              valid JSON (json.JSONDecodeError).
    """
    try:
        debug(
            "JSON loading started",
            extra={
                "path": str(path),
                "wrap_enabled": wrap,
                "context": "JSON loading",
            },
        )

        # Load JSON data from file at
        # specified path
        with open(path) as f:
            json_data = json.load(f)

        debug(
            "JSON loading completed",
            extra={
                "path": str(path),
                "keys_num": (
                    len(json_data) if isinstance(json_data, dict) else None
                ),
                "wrap_enabled": wrap,
                "context": "JSON loading",
            },
        )

        # Check whether to wrap JSON
        # data into a Box object, allowing
        # dot-notation access
        if wrap:
            return Box(json_data)
        return json_data
    except (OSError, json.JSONDecodeError) as e:
        msg = "JSON loading failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "wrap_enabled": wrap,
                "context": "JSON loading",
            },
        )
        raise RuntimeError(msg) from e
