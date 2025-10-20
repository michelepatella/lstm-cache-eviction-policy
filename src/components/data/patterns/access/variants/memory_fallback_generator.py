from typing import List

import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def generate_memory_fallback_pattern(
    memory_interval: int,
    memory_offset: int,
    requests: List[int],
    requests_count: int,
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
) -> int:
    """
    Determine the next key to be accessed according to the memory/fallback pattern.

    This function simulates accesses based on a memory interval, or zipfian
    accesses. If the current request index matches the memory  interval and
    enough history exists, it selects a key previously accessed memory offset
    steps back. Otherwise, it  selects a key randomly according to the provided
    Zipfian probabilities.

    Args:
        memory_interval (int): Number of requests between memory-based accesses.
        memory_offset (int): Steps back in request history to pick the key.
        requests (List[int]): List of keys requested so far.
        requests_count (int): Number of requests generated so far.
        keys_range (np.ndarray): List of available keys.
        zipf_probs (np.ndarray): List of normalized Zipfian probabilities
                                 corresponding to keys range.

    Returns:
        int: Index of the next accessed key.

    Raises:
        RuntimeError: If generating the memory/fallback pattern fails:
            * Accessing the request history due to empty or too short
              list (IndexError, ValueError).
            * Random selection from Zipfian probabilities due to invalid
              shape or normalization (ValueError).
            * Using invalid arguments or data types (TypeError).
    """
    try:
        debug(
            f"Memory interval for memory/fallback pattern: {memory_interval}"
        )

        # Based on the memory interval
        if (requests_count % memory_interval == 0) and (
            requests_count > memory_offset
        ):
            # Requested key as the one requested
            # memory offset steps back in the history
            requested_key = requests[-memory_offset]
            debug(
                f"Requested key selected looking "
                f"at {memory_offset} steps back in the history"
            )
        else:
            # Determine the requested key randomly,
            # according to Zipfian key probabilities
            requested_key = np.random.choice(keys_range, p=zipf_probs)
            debug("Requested key selected by fallback pattern")

        debug(f"(Memory/fallback pattern) Key requested: {requested_key}")

        return requested_key
    except (IndexError, TypeError, ValueError) as e:
        msg = "Failed to generate memory/fallback pattern"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
