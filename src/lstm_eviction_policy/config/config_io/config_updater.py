from typing import Callable

import yaml

from lstm_eviction_policy.config.config_io.config_loader import load_config
from lstm_eviction_policy.config.config_io.config_locator import get_config_abs_path
from lstm_eviction_policy.utils.logs.log_utils import error, info


def _merge_config(original_config: dict | None, updated_config: dict) -> dict:
    """
    Recursively merge an update configuration object
    into the original configuration object.

    This function takes an original configuration object
    and an update one, and merges them recursively. If a
    key exists in both objects, the merge is applied recursively.
    Otherwise, the update value replaces the original one.

    Parameters:
        original_config (dict | None): Original configuration object. If none, an empty
            dictionary is used.
        updated_config (dict): Configuration object containing updates to
            apply to the original configuration object.

    Returns:
        dict: Updated configuration object after merging.

    Raises:
        RuntimeError: If the updated configuration object cannot be merged
                      into the original one due to excessive recursion depth.
    """
    # Check whether the original
    # configuration is None, using
    # an empty dictionary consequently
    if original_config is None:
        original_config = {}

    # Check whether the updated configuration
    # object is not a dictionary, returning
    # the original configuration object consequently
    if not isinstance(updated_config, dict):
        error("Updated configuration object must be a dictionary")
        raise TypeError("Updated configuration object must be a dictionary")

    try:
        for key, value in updated_config.items():
            # If the current value is a dictionary
            # and is contained in both objects, apply
            # merge recursively
            if isinstance(value, dict) and isinstance(original_config.get(key), dict):
                original_config[key] = _merge_config(original_config[key], value)
            else:
                # Otherwise, extract the
                # corresponding value
                original_config[key] = value
    except RecursionError as e:
        error(f"Failed to merge configuration objects: {e}")
        raise RuntimeError("Failed to merge configuration objects") from e

    info(f"Configuration objects merged")

    return original_config


def update_config(updated_config: dict, prepare_config: Callable) -> dict:
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

    # Load the original YAML configuration file
    original_config = load_config()

    # Merge updated configuration object into
    # the original one, and get the resulting
    # configuration object
    merged_config = _merge_config(original_config, updated_config)

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
        error(f"Failed to update YAML configuration file at {abs_config_path}: {e}")
        raise RuntimeError(
            f"Failed to update YAML configuration file at {abs_config_path}"
        ) from e

    # Revalidate the updated YAML configuration
    # file and get the new settings (if everything went well)
    new_config_settings = prepare_config()

    info(f"YAML configuration file updated at {abs_config_path}")

    return new_config_settings
