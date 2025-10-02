from typing import Callable

import yaml

from pipeline.config.classes.Config import ConfigDict
from pipeline.config.config_io.config_loader import load_config
from pipeline.config.config_io.update.config_merger import merge_config
from pipeline.config.config_io.utils.config_locator import get_config_abs_path
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


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
                merged_config,  # New configuration object
                config_file,  # Previous configuration object
                default_flow_style=False,  # Use block style for YAML file
                sort_keys=False,  # Preserve the original order of the keys
                allow_unicode=True,  # Allow writing Unicode characters to YAML file
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
