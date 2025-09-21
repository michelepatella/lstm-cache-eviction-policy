from typing import Any

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


def get_config_param(config: dict, key: str) -> Any:
    """
    Retrieve a configuration parameter from
    configuration object.

    This function retrieves a configuration parameter —
    given its key — from a configuration object containing
    all the parameters required to run the pipeline.

    Parameters:
        config (dict): Configuration object.
        key (str): Key of the configuration parameter to be retrieved.

    Returns:
        Any: Value of the configuration parameter corresponding to the specified key.

    Raises:
        RuntimeError: If the requested parameter value cannot be retrieved from
                      the configuration object due to unexistent or invalid key.
    """
    # Default: the key of the parameter
    # to be retrieve is composed by a single key (itself)
    keys = [key]

    # Extract all the possible sub-keys contained
    # by the specified key (e.g., "data.distribution.seed")
    # is split into [data, distribution, seed]
    if isinstance(key, str):
        keys = key.split(".")

    # Initialize the dictionary where
    # to look for the next key's value,
    # which will be finally set equal to
    # the value of parameter to be retrieved
    value = config

    debug(
        f"Configuration parameter to be retrieved "
        f"from YAML configuration file: {key}"
    )

    try:
        # Retrieve value of specified parameter
        # starting from the outermost dictionary (e.g., data)
        # to the innermost one (e.g., seed)
        for subkey in keys:
            value = value[subkey]
            debug(f"Traversing key '{subkey}', intermediate value: {value}")
    except (KeyError, TypeError) as e:
        msg = f"Failed to retrieve configuration parameter '{key}' from configuration object"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Configuration parameter '{key}' value retrieved: {value}")

    return value
