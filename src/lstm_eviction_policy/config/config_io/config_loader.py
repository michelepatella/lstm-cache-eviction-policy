import yaml
from yaml import YAMLError

from lstm_eviction_policy.config.config_io.config_locator import \
    get_config_abs_path
from lstm_eviction_policy.utils.logs.log_utils import error, info


def load_config() -> dict:
    """
    Load the YAML configuration file for
    the whole pipeline.

    This function reads a YAML configuration file containing
    all parameters required to run the entire pipeline.

    Returns:
        dict: Configuration object with all parameters.

    Raises:
        RuntimeError: If the YAML configuration file cannot be loaded
                      due to file errors or YAML parsing issues.
    """
    # Get the absolute path of
    # the YAML configuration file
    abs_config_path = get_config_abs_path()

    try:
        # Load the YAML configuration
        # file from its absolute path
        with open(abs_config_path, "r") as f:
            config_file = yaml.safe_load(f)
    except (
        OSError,
        YAMLError,
    ) as e:
        error(f"Failed to load YAML configuration file at {abs_config_path}: {e}")
        raise RuntimeError(
            f"Failed to load YAML configuration file at {abs_config_path}"
        ) from e

    info(f"YAML configuration file loaded from {abs_config_path}")

    return config_file
