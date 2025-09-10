from pathlib import Path

from src.lstm_eviction_policy.utils.logs.log_utils import error, info


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
        abs_config_path = str(Path(__file__).resolve().parents[3] / "config.yaml")
    except (NameError, TypeError, AttributeError, OSError) as e:
        error(
            f"Failed to resolve YAML configuration absolute path from {__file__}: {e}"
        )
        raise RuntimeError(
            f"Failed to resolve YAML configuration absolute path from {__file__}"
        ) from e

    info(f"YAML configuration file absolute path resolved ({abs_config_path})")

    return abs_config_path
