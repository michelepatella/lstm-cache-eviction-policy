from typing import Any, Dict, List

from config.classes.Config import Config
from pipeline.validation.search_space.combinations.utils.all_sections_combinator import (
    get_all_sections_combinations,
)
from pipeline.validation.search_space.combinations.utils.single_section_combinator import (
    get_single_section_combinations,
)
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def get_parameters_combination(
    config: Config,
) -> List[Dict[str, Any]]:
    """
    Generate all possible parameter combinations from the search space.

    This function flattens each section of the search space, generates
    all possible combinations of parameters, reconstructs the nested
    dictionary structure for each combination, and returns the list of
    fully nested parameter configurations.

    Parameters:
        config (Config): Configuration object.

    Returns:
        List[Dict[str, Any]]: List of nested dictionaries
                              representing all parameter combinations.

    Raises:
        RuntimeError: If an error occurs while
                      generating parameter combinations, e.g.:
            * Expected keys are missing in the search space.
            * Search space or keys are of invalid type.
            * No parameter combinations could be generated.
    """
    try:
        section_combinations = []

        # Generate combinations per section
        # and save them
        search_space = config.validation.search_space.model_dump()
        for section, params_dict in search_space.items():
            section_values = get_single_section_combinations(
                section, params_dict
            )
            section_combinations.append((section, section_values))

        # Combine all sections into
        # final parameter configurations
        param_combinations = get_all_sections_combinations(
            section_combinations
        )

        info("Parameter combinations generated")

        return param_combinations
    except (KeyError, TypeError, ValueError) as e:
        msg = "Failed to generate parameter combinations"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
