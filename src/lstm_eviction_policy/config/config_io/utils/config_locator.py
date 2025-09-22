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
    debug(
        f"YAML configuration file absolute path"
        f" resolution from __file__: {__file__}"
    )

    try:
        # Resolve the absolute path of
        # YAML configuration file
        resolved_path = Path(__file__).resolve()
        abs_config_path = str(resolved_path.parents[5] / "config.yaml")
        debug(
            f"Parent directories used to resolve absolute path of YAML configuration file: "
            f"{[p for p in resolved_path.parents[:5]]}"
        )
    except (NameError, TypeError, AttributeError, OSError) as e:
        msg = f"Failed to resolve YAML configuration absolute path from {__file__}"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"YAML configuration file absolute path resolved ({abs_config_path})")

    return abs_config_path
