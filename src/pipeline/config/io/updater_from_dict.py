from typing import Any, Dict

from pipeline.config.classes.Config import Config
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def update_config_from_dict(
    config: Config, updates: Dict[str, Any], path: str = None
) -> None:
    """
    Update configuration object from a dictionary recursively.

    This function traverses the provided dictionary and applies
    its values to the corresponding fields in the configuration
    object recursively.

    Args:
        config (Config): Configuration object to update.
        updates (Dict[str, Any]): Dictionary of updates to apply.
        path (str): Dot-separated path for logging purposes.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while updating configuration object, e.g.:
            * Section not found in configuration.
            * Field in configuration section not found.
            * Type or attribute errors during assignment.
    """
    try:
        # Build full path through keys
        # and values, using dot notation
        for key, value in updates.items():
            full_path = f"{path}.{key}" if path else key

            # If the value is a dictionary, recurse into it
            if isinstance(value, dict):
                # Check whether the section does
                # not exist in the configuration
                if not hasattr(config, key):
                    msg = f"Section '{full_path}' not found in configuration"
                    error("%s", msg)
                    raise KeyError(msg)

                # Apply updates recursively
                update_config_from_dict(getattr(config, key), value, full_path)
            else:
                # Check whether the field does
                # not exist in the configuration
                if not hasattr(config, key):
                    msg = f"Field '{full_path}' not found in configuration"
                    error("%s", msg)
                    raise KeyError(msg)

                # If it exists, update it
                setattr(config, key, value)
                debug(f"'{full_path}' updated -> {value}")

        info("Configuration updated from dictionary")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        msg = "Failed to update configuration from dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
