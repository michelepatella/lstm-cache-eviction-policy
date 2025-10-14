from typing import Any, Dict
import yaml

from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Save a dictionary as a YAML file.

    This function saves the provided dictionary
    into a YAML file at the specified path.

    Args:
        data (Dict[str, Any]): Dictionary to save as YAML.
        path (str): Path where the YAML file will be saved.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while saving YAML file e.g.:
                        * YAML file cannot be written due to
                          operating system errors.
                        * YAML file cannot be written due to
                          permission issues.
    """
    try:
        # Save provided data dictionary to
        # specified path
        with open(path, "w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
            )
    except OSError as e:
        msg = "Failed to save YAML file"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"YAML file saved to {path}")
