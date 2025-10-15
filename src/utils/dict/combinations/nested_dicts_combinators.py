import itertools
from typing import Any, Dict, List

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.dict.flattener import flatten_dict
from utils.dict.value_setter import set_dict_value


def combine_nested_dicts(nested_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations from a nested dictionary.

    Each leaf value in the dictionary can be a list of options. This function
    computes the Cartesian product of all possible leaf values and reconstructs
    nested dictionaries for each combination.

    Args:
        nested_dict (Dict[str, Any]): Nested dictionary with leaf values that are
                                      either single values or lists of options.

    Returns:
        List[Dict[str, Any]]: List of nested dictionaries representing all possible
                              combinations of values.

    Raises:
        RuntimeError: If an error occurs while generating combinations, e.g.:
            * Input is not a dictionary or has unexpected types.
            * Leaf values cannot be iterated or deep copied.
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
        combination_dicts: List[Dict[str, Any]] = []
        for values in all_combinations:
            combo: Dict[str, Any] = {}
            for key_path, value in zip(keys, values):
                set_dict_value(combo, key_path, value)
            combination_dicts.append(combo)

        debug(f"{len(combination_dicts)} combinations generated from nested dictionary")
        info("Nested dictionary combinations generated")

        return combination_dicts
    except (TypeError, ValueError) as e:
        msg = "Failed to generate combinations from nested dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e