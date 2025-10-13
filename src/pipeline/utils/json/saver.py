import json
from typing import Dict, List

from const import JSON_INDENT
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def save_json(data_dict: Dict | List[Dict], path: str) -> None:
    """
    Save a data dictionary as a JSON file.

    This function saves a data dictionary as a
    JSON file to the provided path.

    Args:
        data_dict (Dict | List[Dict]): Data to save.
        path (str): File path where the JSON will be saved.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while saving JSON file e.g.:
            * If a value of data dictionary cannot be
              serialized to JSON.
            * If an operating system error occurs while
              JSON file.
    """
    debug(f"Path where to save JSON: {path}")

    try:
        # Save data dictionary as JSON file
        # to the specified path
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=JSON_INDENT)
    except (TypeError, OSError) as e:
        msg = "Failed to save JSON"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"JSON saved to {path}")
