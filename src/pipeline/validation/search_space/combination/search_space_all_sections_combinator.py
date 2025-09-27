import copy
import itertools
from typing import Any, Dict, List, Tuple

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def get_all_sections_combinations(
    section_combinations: List[Tuple[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine parameter combinations from all
    sections into full configurations.

    Parameters:
        section_combinations (List[Tuple[str, Any]]):
            Section name and its combinations.

    Returns:
        List[Dict[str, Any]]: List of fully combined parameter dictionaries.

    Raises:
        ValueError: If no full combinations could be generated.
        RuntimeError: If an error while combining section parameters occurs.
    """
    try:
        # Generate all possible combinations
        # across sections
        all_combos = list(
            itertools.product(*[vals for _, vals in section_combinations])
        )

        # Build the final list of full parameter
        # dictionaries, each one of them containing
        # all sections with one combination each
        param_combinations = []
        for combo in all_combos:
            full_dict = {}
            for (section, _), section_dict in zip(section_combinations, combo):
                full_dict[section] = copy.deepcopy(section_dict)
            param_combinations.append(full_dict)

        debug(
            f"Total parameter combinations across all sections: {len(param_combinations)}"
        )

        # Check whether no parameter
        # combination has been found
        if not param_combinations:
            msg = "No all sections parameter combinations generated"
            error("%s", msg)
            raise ValueError(msg)

        info("All section combinations generated")

        return param_combinations
    except ValueError as e:
        msg = "Failed to generate all sections parameter combinations"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
