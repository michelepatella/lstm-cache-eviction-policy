from typing import Dict, List

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.logs.log_utils import error, info
from lstm_eviction_policy.validation.search_space.combination.search_space_all_sections_combinator import (
    get_all_sections_combinations,
)
from lstm_eviction_policy.validation.search_space.combination.search_space_section_combinator import (
    get_section_combinations,
)


def get_parameters_combination(
    config: Config,
) -> List[Dict[str, List[int | float]]]:
    """
    Generate all possible parameter combinations from the search space.

    This function flattens each section of the search space, generates
    all possible combinations of parameters, reconstructs the nested
    dictionary structure for each combination, and returns the list of
    fully nested parameter configurations.

    Parameters:
        config (Config): Configuration object.

    Returns:
        List[Dict[str, List[int | float]]]: List of nested dictionaries
                                            representing all parameter combinations.

    Raises:
        KeyError: If expected keys are missing in the search space.
        TypeError: If search space or keys are of invalid type.
        ValueError: If no parameter combinations could be generated.
    """
    try:
        section_combinations = []

        # Generate combinations per section
        # and save them
        for section, params_dict in config.search_space.items():
            section_values = get_section_combinations(section, params_dict)
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
