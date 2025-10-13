import yaml
from yaml import YAMLError

from pipeline.config.classes.Config import ConfigDict
from pipeline.config.io.utils.locator import get_config_abs_path
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def load_config() -> ConfigDict:
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

    debug(
        f"YAML configuration file absolute path "
        f"to be loaded: {abs_config_path}"
    )

    try:
        # Load the YAML configuration
        # file from its absolute path
        with open(abs_config_path, "r") as f:
            config_file = yaml.safe_load(f)
            debug(
                f"YAML configuration file content"
                f" type loaded: {type(config_file)}"
            )
    except (
        OSError,
        YAMLError,
    ) as e:
        msg = "Failed to load YAML configuration"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"YAML configuration file loaded from {abs_config_path}")

    return config_file
