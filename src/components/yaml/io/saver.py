from typing import Any, Dict

import yaml

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def save_yaml(data_dict: Dict[str, Any], path: str) -> None:
    """
    Save a data dictionary as a YAML file.

    This function saves the provided data dictionary
    as a YAML file at the specified path.

    Args:
        data_dict (Dict[str, Any]): Data dictionary to save as YAML.
        path (str): Path to save YAML file at.

    Returns:
        None

    Raises:
        RuntimeError: If saving YAML file fails:
            * File cannot be written due to operating system errors (OSError).
            * File cannot be written due to permission issues (OSError).
    """
    try:
        debug(f"Path to save YAML file at: {path}")

        # Save provided data dictionary at
        # specified path
        with open(path, "w") as f:
            yaml.dump(data_dict, f)

        info(f"YAML file saved to: {path}")
    except OSError as e:
        msg = "Failed to save YAML file"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
