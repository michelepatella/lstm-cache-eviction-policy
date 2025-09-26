import itertools
from typing import Dict, List

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info
from lstm_eviction_policy.validation.search_space.searc_space_setter import (
    set_nested_dict,
)
from lstm_eviction_policy.validation.search_space.search_space_flattener import (
    flatten_search_space,
)


def get_section_combinations(
    section: str, params_dict: Dict[str, List[int | float]]
) -> List[Dict[str, List[int | float]]]:
    """
    Generate all possible parameter combinations for a single section.

    Parameters:
        section (str): Section name in the search space.
        params_dict (Dict[str, List[int | float]]): Section parameters.

    Returns:
        List[Dict[str, List[int | float]]]: All possible parameter combinations
                                            for the section.

    Raises:
        TypeError: If params_dict is not iterable.
        ValueError: If no combination could be generated.
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
            combo = {}
            for key_path, value in zip(keys, values):
                # Set nested key-value pairs to reconstruct
                # original structure
                set_nested_dict(combo, key_path, value)
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
