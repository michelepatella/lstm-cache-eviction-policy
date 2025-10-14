from typing import Dict, Any

import yaml
from yaml import YAMLError

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file from path.

    This function reads a YAML file from the specified path.

    Args:
        path (str): Path to load YAML file from.

    Returns:
        Dict[str, Any]: YAML file loaded.

    Raises:
        RuntimeError: If the YAML file cannot be loaded
                      due to file errors or YAML parsing issues.
    """
    debug(
        f"YAML file path to be loaded: {path}"
    )

    try:
        # Load the YAML file from its path
        with open(path, "r") as f:
            yaml_file = yaml.safe_load(f)
    except (
        OSError,
        YAMLError,
    ) as e:
        msg = "Failed to load YAML file"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"YAML file loaded from {path}")

    return yaml_file
