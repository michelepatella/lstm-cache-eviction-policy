import numpy as np

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info


def generate_cycle_pattern(
    cycle_base: int,
    cycle_divisor: int,
    cycle_mod: int,
    requests_count: int,
    keys_range: np.ndarray,
) -> int:
    """
    Determine the next key to be accessed
    according to cycle pattern.

    This function determines which is the next
    key to be accessed according to cycle
    pattern. This pattern simulates cyclic accesses
    to a subset of keys dynamically built, depending
    on the number of requests already generated.

    Parameters:
        cycle_base (int): Minimum number of keys in the cycle.
        cycle_divisor (int): Divisor controlling how quickly the
                             cycle length increases.
        cycle_mod (int): Modulo that limits the growth of the
                         cycle length.
        requests_count (int): Number of requests generated so far.
        keys_range (np.ndarray): List of available keys.

    Returns:
      int: Index of the next accessed key.
    """
    debug(
        f"Cycle base to determine cycle length in cycle pattern: {cycle_base}"
    )
    debug(
        f"Cycle divisor to determine cycle length in cycle pattern: {cycle_divisor}"
    )
    debug(f"Cycle mod to determine cycle length in cycle pattern: {cycle_mod}")

    # Calculate the cycle length dynamically,
    # based on number of requests generated so far
    # (cycle base ensures minimum cycle length)
    cycle_length = cycle_base + (requests_count // cycle_divisor) % cycle_mod

    debug(f"Cycle length for cycle pattern: {cycle_length}")

    # Select a subset of keys of
    # cycle length size, starting from
    # the head of the keys list
    cycle = keys_range[:cycle_length]

    debug(f"Candidate keys selected for cycle pattern: {cycle}")

    # Determine the requested key
    # within the subset of key just
    # selected, based on the number of
    # requests generated so far as well as
    # the cycle length
    requested_key = int(cycle[requests_count % cycle_length])

    info(f"(Cycle pattern) Key requested: {requested_key}")

    return requested_key
