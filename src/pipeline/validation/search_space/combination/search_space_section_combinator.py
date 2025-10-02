import itertools
from typing import Any, Dict, List

from const import CONFIG_SECTIONS_WITH_PARAMS
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info
from pipeline.validation.search_space.search_space_flattener import (
    flatten_search_space,
)
from pipeline.validation.search_space.search_space_setter import (
    set_nested_dict,
)
from pipeline.validation.search_space.section_params_wrapper import (
    wrap_section_params,
)


def get_section_combinations(
    section: str, params_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate all possible parameter combinations for a single section.

    Parameters:
        section (str): Section name in the search space.
        params_dict (Dict[str, Any]): Section parameters.

    Returns:
        List[Dict[str, Any]]: All possible parameter combinations
                              for the section.

    Raises:
        ValueError: If no combination could be generated.
        RuntimeError: If an error while generating
                      all section parameters combinations
                      occurs, e.g.:
            * If params_dict is not iterable.
    """
    try:
        # Flatten the section
        flat_params = flatten_search_space(params_dict)

        # Extract keys and corresponding values
        keys = [key for key, _ in flat_params]
        value_lists = [v for _, v in flat_params]

        # Generate all possible combinations
        combinations = list(itertools.product(*value_lists))

        # Build all possible parameter dictionaries
        # for the current section
        section_values = []
        for values in combinations:
            combo: Dict[str, Any] = {}
            for key_path, value in zip(keys, values):
                # Set nested key-value pairs to reconstruct
                # original structure
                set_nested_dict(combo, key_path, value)

            # Wrap section parameters ensuring
            # consistent representation with respect to
            # those of configuration object
            combo = wrap_section_params(
                combo, section, CONFIG_SECTIONS_WITH_PARAMS
            )

            section_values.append(combo)

        debug(
            f"{len(combinations)} combinations generated for section {section}"
        )

        # Check whether section combination
        # is empty
        if not section_values:
            msg = "No parameter combinations found"
            error("%s", msg)
            raise ValueError(msg)

        info(f"Section combinations generated for {section}")

        return section_values
    except (TypeError, ValueError) as e:
        msg = "Failed to generate section combinations"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
