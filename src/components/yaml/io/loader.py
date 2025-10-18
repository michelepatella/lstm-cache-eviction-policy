from typing import Any, Dict

import yaml
from yaml import YAMLError

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file from path.

    This function reads a YAML file from the specified path.

    Args:
        path (str): Path to load YAML file from.

    Returns:
        Dict[str, Any]: YAML file loaded.

    Raises:
        RuntimeError: If YAML file loading fails:
            * File cannot be accessed or read (OSError).
            * YAML parsing fails due to invalid format (YAMLError).
    """
    try:
        debug(f"Path to load YAML file from: {path}")

        # Load the YAML file from its path
        with open(path, "r") as f:
            yaml_file = yaml.safe_load(f)

        info(f"YAML file loaded from: {path}")

        return yaml_file
    except (
        OSError,
        YAMLError,
    ) as e:
        msg = "Failed to load YAML file"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
