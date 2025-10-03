from typing import Any, Dict

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def wrap_section_params(
    combo: Dict[str, Any], section: str, wrapper_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Wrap the section dictionary with a specified
    key if required.

    This function ensures that the section structure
    matches the Pydantic BaseModel, handling both top-level
    and nested sections.

    Parameters:
        combo (Dict[str, Any]): Parameter combination
                                dictionary for the section.
        section (str): Section name (top-level or nested).
        wrapper_map (Dict[str, str]): Mapping of section
                                      names to wrapper keys.

    Returns:
        Dict[str, Any]: The modified parameter dictionary with wrapper applied.

    Raises:
        RuntimeError: If an error occurs during wrapping, e.g.:
            * Missing keys.
            * Invalid types.
    """
    debug(f"Applying params wrapper for section '{section}'")

    try:
        if section in wrapper_map:
            wrapper = wrapper_map[section]

            # Handle nested sections
            if "." in section:
                keys = section.split(".")
                target = combo

                # Traverse nested dictionary levels
                for k in keys[:-1]:
                    if k not in target or not isinstance(target[k], dict):
                        msg = (
                            f"Nested key '{k}' not found "
                            f"or not a dictionary in combo"
                        )
                        error("%s", msg)
                        raise KeyError(msg)
                    target = target[k]

                # Wrap the last key
                if keys[-1] not in target or not isinstance(
                    target[keys[-1]], dict
                ):
                    msg = (
                        f"Final nested key '{keys[-1]}' "
                        f"not found or not a dictionary in combo"
                    )
                    error("%s", msg)
                    raise KeyError(msg)
                target[keys[-1]] = {wrapper: target[keys[-1]]}

            else:
                # Top-level section
                if not isinstance(combo, dict):
                    msg = (
                        f"Expected combo to be a dictionary, got {type(combo)}"
                    )
                    error("%s", msg)
                    raise TypeError(msg)
                combo = {wrapper: combo}

        debug(f"Params wrapper applied for section '{section}': {combo}")
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        msg = "Failed to apply params wrapper"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Section '{section}' processed with params wrapper")
    return combo
