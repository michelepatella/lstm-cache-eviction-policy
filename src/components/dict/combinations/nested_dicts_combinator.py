import itertools
from typing import Any, Dict, List

from components.dict.operations.flattener import flatten_dict
from components.dict.operations.value_setter import set_dict_value
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def combine_nested_dicts(nested_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations from a nested dictionary.

    This function computes the Cartesian product of all possible leaf values
    in the dictionary and reconstructs nested dictionaries for each combination.

    Args:
        nested_dict (Dict[str, Any]): Nested dictionary with leaf values that are
                                      either single values or lists of options.

    Returns:
        List[Dict[str, Any]]: List of nested dictionaries representing all possible
                              combinations of values.

    Raises:
        RuntimeError: If generating combinations from the nested dictionary fails:
            * Input is not a dictionary (TypeError).
            * Leaf values cannot be iterated or copied properly (TypeError).
            * Cartesian product computation fails due to invalid input structure
              (ValueError).
            * Setting values in nested dictionaries fails due to incorrect key paths
              or incompatible types (TypeError).
    """
    try:
        # Flatten the dictionary to get key paths and values
        flat_items = flatten_dict(nested_dict)
        keys = [key_path for key_path, _ in flat_items]
        value_lists = [
            v if isinstance(v, list) else [v] for _, v in flat_items
        ]

        # Compute Cartesian product of all leaf value options
        all_combinations = list(itertools.product(*value_lists))

        # Reconstruct nested dictionaries for each combination
        combination_dicts = []
        for values in all_combinations:
            combo = {}
            for key_path, value in zip(keys, values):
                set_dict_value(combo, key_path, value)
            combination_dicts.append(combo)

        info(
            f"Nested dictionary combinations generated "
            f"({len(combination_dicts)} combinations)"
        )

        return combination_dicts
    except (TypeError, ValueError) as e:
        msg = "Failed to generate combinations from nested dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
