from typing import Any, Dict

from box.box import Box

from config.classes.Config import Config
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def save_best_params(best_params: Dict[str, Any], config: Config) -> Config:
    """
    Save the best parameters to the configuration.

    This function updates the provided configuration
    object with the best parameters found during optimization.

    Parameters:
        best_params (Dict[str, Any]): Best parameters found.
        config (Config): Current configuration object.

    Returns:
        Config: Updated configuration object.
    """
    debug(f"Best params to save: {best_params}")

    try:

        def _apply_updates(
            obj: Any, updates: Dict[str, Any], path: str = ""
        ) -> None:
            """
            Recursively apply updates to a Pydantic model or dictionary.

            Parameters:
                obj (Any): The current Pydantic model or dictionary to update.
                updates (Dict[str, Any]): Dictionary of updates to apply.
                path (str): Dot-separated path for logging purposes.

            Returns:
                None

            Raises:
                RuntimeError: If an error occurs during parameter saving, e.g.:
                    * Section not found in configuration.
                    * Field in section not found.
            """
            # Build full path through keys
            # and values
            for key, value in updates.items():
                full_path = f"{path}.{key}" if path else key

                # If the value is a dictionary, recurse into it
                if isinstance(value, dict):
                    # Check whether the current object
                    # has not the specified key as attribute
                    if not hasattr(obj, key):
                        msg = (
                            f"Section '{full_path}' not found in configuration"
                        )
                        error("%s", msg)
                        raise KeyError(msg)

                    # Apply updates recursively
                    _apply_updates(getattr(obj, key), value, full_path)
                else:
                    # Update the actual value if field exists
                    if not hasattr(obj, key):
                        msg = f"Field '{full_path}' not found in configuration"
                        error("%s", msg)
                        raise KeyError(msg)

                    # If it exists, update it
                    setattr(obj, key, value)
                    debug(f"(Best params) '{full_path}' updated -> {value}")

        # Use Box for best params
        best_params_box = Box(best_params)

        # Apply updates recursively
        _apply_updates(config, best_params_box.to_dict())

        info("Saved best parameters in configuration")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        msg = "Failed to save best parameters in configuration"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    return config
