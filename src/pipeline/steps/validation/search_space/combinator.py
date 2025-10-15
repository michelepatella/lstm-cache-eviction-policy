from typing import Any, Dict, List

from pipeline.config.classes.Config import Config
from utils.dict.combinations.nested_dicts_combinators import combine_nested_dicts
from utils.dict.combinations.nested_dict_lists_combinator import combine_nested_dict_lists
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

    Args:
        config (Config): Configuration object.

    Returns:
        List[Dict[str, Any]]: List of nested dictionaries
                              representing all parameter combinations.

    Raises:
        RuntimeError: If an error occurs while
                      generating parameter combinations, e.g.:
            * Search space or keys are of invalid type.
            * No parameter combinations could be generated.
    """
    try:
        section_combination_dicts = []

        # Generate combinations per section
        # and save them
        search_space = config.validation.search_space.model_dump()
        for section, params_dict in search_space.items():
            section_values = combine_nested_dicts(params_dict)
            section_combination_dicts.append((section, section_values))

        # Combine all sections into
        # final parameter configurations
        param_combinations = combine_nested_dict_lists(section_combination_dicts)

        info("Parameter combinations generated")

        return param_combinations
    except (TypeError, ValueError) as e:
        msg = "Failed to generate parameter combinations"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
