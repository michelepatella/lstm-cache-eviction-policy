from lstm_eviction_policy.config.classes.Config import (
    ConfigDict,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def merge_config(
    original_config: ConfigDict | None,
    updated_config: ConfigDict,
) -> ConfigDict:
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
        TypeError: If the updated configuration objects is not
                   a dictionary.
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
        msg = "Updated configuration object must be a dictionary"
        error("%s", msg)
        raise TypeError(msg)

    try:
        for key, value in updated_config.items():
            # If the current value is a dictionary
            # and is contained in both objects, apply
            # merge recursively
            if isinstance(value, dict) and isinstance(
                original_config.get(key), dict
            ):
                debug(f"Merging nested key '{key}'")
                original_config[key] = merge_config(
                    original_config[key],
                    value,
                )
            else:
                debug(f"Merging key '{key}': {value}")
                # Otherwise, extract the
                # corresponding value
                original_config[key] = value
    except RecursionError as e:
        msg = "Failed to merge configuration objects"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Configuration objects merged")

    return original_config
