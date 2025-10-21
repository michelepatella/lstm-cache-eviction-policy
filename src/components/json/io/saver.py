import json
from typing import Dict, List, Union

from components.const import JSON_INDENT
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_json(
    data_dict: Union[Dict, List[Dict]],
    path: str,
    json_indent: int = JSON_INDENT,
) -> None:
    """
    Save a data dictionary as a JSON file.

    This function saves a data dictionary as a
    JSON file at the specified path.

    Args:
        data_dict (Union[Dict, List[Dict]]): Data to save.
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
        debug(f"Path to save JSON to: {path}")

        # Save data dictionary as JSON file
        # to the specified path
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=json_indent)

        debug(f"JSON saved to: {path}")
    except (TypeError, OSError) as e:
        msg = "Failed to save JSON"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
