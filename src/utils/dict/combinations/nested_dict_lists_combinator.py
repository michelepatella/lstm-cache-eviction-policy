import copy
import itertools
from typing import Any, Dict, List

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def combine_nested_dict_lists(dict_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Combine multiple lists of nested dictionaries into all possible full combinations.

    Each list contains dictionaries representing a set of options.
    This function computes the Cartesian product across all lists and merges
    one dictionary from each list into a single dictionary per combination.

    Args:
        dict_lists (List[List[Dict[str, Any]]]): List of lists of nested dictionaries.

    Returns:
        List[Dict[str, Any]]: List of merged nested dictionaries for all possible combinations.

    Raises:
        RuntimeError: If an error occurs while combining dictionaries, e.g.:
            * Input is not a list of lists of dictionaries.
            * One of the dictionaries has unexpected types that prevent deep copying.
    """
    try:
        # Compute Cartesian product across all dictionary lists
        all_combos = list(itertools.product(*dict_lists))

        combined_dicts: List[Dict[str, Any]] = []
        for combo_tuple in all_combos:
            merged: Dict[str, Any] = {}
            for d in combo_tuple:
                for k, v in d.items():
                    # Deep copy to avoid mutating
                    # input dictionaries
                    merged[k] = copy.deepcopy(v)
            combined_dicts.append(merged)

        debug(f"Total combined nested dictionaries lists: {len(combined_dicts)}")
        info("Nested dictionary lists combined")

        return combined_dicts
    except (TypeError, ValueError) as e:
        msg = "Failed to combine nested dictionary lists"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e