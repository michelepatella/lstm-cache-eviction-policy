from typing import Callable

import yaml

from config.classes.Config import ConfigDict
from config.io.loader import load_config
from config.io.update.merger import merge_config
from config.io.utils.locator import get_config_abs_path
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def update_config(
    updated_config: ConfigDict,
    prepare_config: Callable,
) -> ConfigDict:
    """
    Update the YAML configuration file by merging the
    updated configuration object into the original one.

    This function takes an updated configuration object and merges
    it into the original one. Then, runs validation on the entire new
    configuration object before using it as new settings.

    Parameters:
        updated_config (dict): Updated configuration object.
        prepare_config (Callable): Method to run validation on
            the new configuration object and set it as new settings.

    Returns:
        dict: Updated, validated configuration object.

    Raises:
        RuntimeError: If the YAML configuration file cannot be updated
                      due to operating system errors.
    """
    # Get the absolute path of the YAML
    # configuration file
    abs_config_path = get_config_abs_path()

    debug(
        f"YAML configuration absolute path where "
        f"to update configuration: {abs_config_path}"
    )

    # Load the original YAML configuration file
    original_config = load_config()

    # Merge updated configuration object into
    # the original one, and get the resulting
    # configuration object
    merged_config = merge_config(original_config, updated_config)

    try:
        # Update the YAML configuration file by
        # overwriting it with the updated version
        with open(abs_config_path, "w") as config_file:
            yaml.dump(
                # New configuration object
                merged_config,
                # Previous configuration object
                config_file,
                # Use block style for YAML file
                default_flow_style=False,
                # Preserve the original order of the keys
                sort_keys=False,
                # Allow writing Unicode characters to YAML file
                allow_unicode=True,
            )
    except OSError as e:
        msg = "Failed to update YAML configuration file"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Revalidate the updated YAML configuration
    # file and get the new settings (if everything went well)
    updated_config = prepare_config()

    info(f"YAML configuration file updated at {abs_config_path}")

    return updated_config
