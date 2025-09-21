from pathlib import Path

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


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
    debug(f"YAML configuration file absolute path resolution from __file__: {__file__}")

    try:
        # Resolve the absolute path of
        # YAML configuration file
        abs_config_path = str(Path(__file__).resolve().parents[4] / "config.yaml")
        debug(
            f"Parent directories used to access YAML configuration file: {[p for p in Path(__file__).resolve().parents[:4]]}"
        )
    except (NameError, TypeError, AttributeError, OSError) as e:
        error(
            f"Failed to resolve YAML configuration absolute path from {__file__}: {e}"
        )
        raise RuntimeError(
            f"Failed to resolve YAML configuration absolute path from {__file__}"
        ) from e

    info(f"YAML configuration file absolute path resolved ({abs_config_path})")

    return abs_config_path
