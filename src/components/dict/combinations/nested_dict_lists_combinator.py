import copy
import itertools
from typing import Any, Dict, List

from components.logs.levels.error_logger import error


def combine_nested_dict_lists(
    dict_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Combine multiple lists of nested dictionaries into all possible
    full combinations.

    Each list contains dictionaries representing a set of options.
    This function computes the Cartesian product across all lists and merges
    one dictionary from each list into a single dictionary per combination.

    Args:
        dict_lists (List[List[Dict[str, Any]]]): List of lists of nested dictionaries.

    Returns:
        List[Dict[str, Any]]: List of merged nested dictionaries for all possible
                              combinations.

    Raises:
        RuntimeError: If combination of nested dictionary lists fails:
            * Input is not a list of lists of dictionaries (TypeError).
            * Deep copying or merging dictionaries fails due to unsupported
              types (TypeError).
            * Cartesian product computation fails due to invalid input structure
              (ValueError).
    """
    try:
        # Compute Cartesian product across all dictionary lists
        all_combos = list(itertools.product(*dict_lists))

        combined_dicts = []
        for combo_tuple in all_combos:
            merged = {}
            for d in combo_tuple:
                for k, v in d.items():
                    # Deep copy to avoid mutating
                    # input dictionaries
                    merged[k] = copy.deepcopy(v)
            combined_dicts.append(merged)

        return combined_dicts
    except (TypeError, ValueError) as e:
        msg = "Combining nested dictionary lists failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dict_lists_length": [len(lst) for lst in dict_lists],
                "context": "Nested dictionary lists combination",
            },
        )
        raise RuntimeError(msg) from e
