import numpy as np

from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    info,
)


def generate_repetition_pattern(
    repetition_interval: int,
    repetition_offset: int,
    requests_count: int,
    requests: list[int],
    keys_range: np.ndarray,
    num_keys: int,
) -> int:
    """
    Determine the next key to be accessed
    according to repetition pattern.

    This function determines which is the next
    key to be accessed according to repetition
    pattern. This pattern simulates some keys are
    accessed regularly, while other accesses involve
    keys randomly selected from a subset of initial keys.

    Parameters:
        repetition_interval (int): Integer indicating how many times
                                   there will be a regular key access.
        repetition_offset (int): Integer determining how many steps back
                                 in requests to pick the next key.
        requests_count (int): Number of requests generated so far.
        requests (list[int]): List of requests generated so far.
        keys_range (np.array): List of keys.
        num_keys (int): Total number of keys.

    Returns:
          int: Index of the next accessed key.
    """
    debug(f"Repetition interval for repetition pattern: {repetition_interval}")
    debug(f"Repetition offset for repetition pattern: {repetition_offset}")

    # Based on a repetition interval
    if requests_count % repetition_interval == 0:
        # Regular access to a key
        requested_key = requests[-repetition_offset]

        debug("Regular repetition pattern access")
    else:
        slice_length = max(1, (num_keys // repetition_offset))
        # Random access to a key taken from
        # a subset of initial keys
        requested_key = np.random.choice(keys_range[:slice_length])

        debug(
            f"Random repetition pattern access among the first {slice_length} keys"
        )

    info(f"(Repetition pattern) Key requested: {requested_key}")

    return requested_key
