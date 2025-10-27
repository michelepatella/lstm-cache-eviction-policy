from typing import Any

import yaml
from yaml import YAMLError

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file from path.

    This function reads a YAML file from the specified path.

    Args:
        path (str): Path to load YAML file from.

    Returns:
        dict[str, Any]: YAML file loaded.

    Raises:
        RuntimeError: If YAML file loading fails:
            * File cannot be accessed or read (OSError).
            * YAML parsing fails due to invalid format (YAMLError).
    """
    try:
        debug(
            "YAML file loading started",
            extra={
                "path": str(path),
                "context": "YAML file loading",
            },
        )

        # Load the YAML file from its path
        with open(path) as f:
            yaml_file = yaml.safe_load(f)

        debug(
            "YAML file loading completed",
            extra={
                "path": str(path),
                "keys_loaded": (
                    list(yaml_file.keys())
                    if isinstance(yaml_file, dict)
                    else None
                ),
                "context": "YAML file loading",
            },
        )

        return yaml_file
    except (
        OSError,
        YAMLError,
    ) as e:
        msg = "Loading YAML file failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "context": "YAML file loading",
            },
        )
        raise RuntimeError(msg) from e
