import numpy as np

from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_memory_fallback_pattern(
    memory_interval: int,
    memory_offset: int,
    requests: list[int],
    requests_count: int,
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
) -> int:
    """
    Determine the next key to be accessed
    according to the memory/fallback pattern.

    This function simulates accesses based
    on a memory interval, or zipfian accesses.
    If the current request index matches the memory
    interval and enough history exists, it selects a key
    previously accessed memory offset steps back.
    Otherwise, it selects a key randomly according
    to the provided Zipfian probabilities.

    Parameters:
        memory_interval (int): Number of requests between
                               memory-based accesses.
        memory_offset (int): Steps back in request history to
                             pick the key.
        requests (list[int]): List of keys requested so far.
        requests_count (int): Number of requests generated so far.
        keys_range (np.ndarray): List of available keys.
        zipf_probs (np.ndarray): List of normalized Zipfian probabilities
                                 corresponding to keys range.

    Returns:
        int: Index of the next accessed key.
    """
    debug(f"Memory interval for memory/fallback pattern: {memory_interval}")

    # Based on the memory interval
    if (requests_count % memory_interval == 0) and (requests_count > memory_offset):
        # Requested key as the one requested
        # memory offset steps back in the history
        requested_key = requests[-memory_offset]

        debug(
            f"Requested key selected looking at {memory_offset} steps back in the history"
        )
    else:
        # Determine the requested key randomly,
        # according to Zipfian key probabilities
        requested_key = np.random.choice(keys_range, p=zipf_probs)

        debug(f"Requested key selected by fallback pattern")

    info(f"(Memory/fallback pattern) Key requested: {requested_key}")

    return requested_key
