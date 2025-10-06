from const import CONFIG_FILE_NAME, PROJECT_ROOT
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def get_config_abs_path() -> str:
    """
    Retrieve the absolute path of the
    YAML configuration file.

    This function resolves the absolute path of the YAML
    configuration file — containing all parameters required
    to run the entire pipeline — before returning it.

    Returns:
        str: Absolute path of the YAML configuration file.

    Raises:
        RuntimeError: If the absolute path of YAML configuration
                      file cannot be resolved.
    """
    try:
        # Resolve the absolute path of
        # YAML configuration file
        abs_config_path = PROJECT_ROOT / CONFIG_FILE_NAME

        debug(f"YAML configuration file absolute" f" path: {abs_config_path}")
    except (
        NameError,
        TypeError,
        AttributeError,
        OSError,
    ) as e:
        msg = "Failed to resolve YAML configuration absolute path"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"YAML configuration file absolute path resolved ({abs_config_path})")

    return abs_config_path
