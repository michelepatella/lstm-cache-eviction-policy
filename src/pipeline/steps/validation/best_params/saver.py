from typing import Any, Dict

from box.box import Box

from pipeline.config.classes.Config import Config
from pipeline.config.io.updater_from_dict import update_config_from_dict
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def save_best_params(best_params: Dict[str, Any], config: Config) -> Config:
    """
    Save the best parameters to the configuration.

    This function updates the provided configuration
    object with the best parameters found during optimization.

    Args:
        best_params (Dict[str, Any]): Best parameters found.
        config (Config): Current configuration object.

    Returns:
        Config: Updated configuration object.
    """
    debug(f"Best params to save: {best_params}")

    # Use Box for best params
    best_params_box = Box(best_params)

    # Apply updates to configuration object
    # in order to save best parameters
    update_config_from_dict(config, best_params_box.to_dict())

    info("Saved best parameters in configuration")

    return config
