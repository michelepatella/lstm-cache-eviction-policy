"""cycle_generator.py

Module for generating cycle-based access patterns.

This module provides the `generate_cycle_pattern` function, which
simulates cyclic access to a subset of keys. The cycle length grows
dynamically depending on the number of requests generated, and
the function returns the index of the next key to access.

Functions:
    generate_cycle_pattern(
        cycle_base: int,
        cycle_divisor: int,
        cycle_mod: int,
        requests_count: int,
        keys_range: ndarray
    ) -> int
        Returns the next key index according to a cycle-based pattern.
"""

import numpy as np

from components.logs.levels.error_logger import error


def generate_cycle_pattern(
    cycle_base: int,
    cycle_divisor: int,
    cycle_mod: int,
    requests_count: int,
    keys_range: np.ndarray,
) -> int:
    """Determine the next key to be accessed according to cycle pattern.

    This function determines which is the next key to be accessed according
    to cycle pattern. This pattern simulates cyclic accesses to a subset of
    keys dynamically built, depending on the number of requests already
    generated.

    Args:
        cycle_base (int): Minimum number of keys in the cycle.
        cycle_divisor (int): Divisor controlling how quickly the cycle length
                             increases.
        cycle_mod (int): Modulo that limits the growth of the cycle length.
        requests_count (int): Number of requests generated so far.
        keys_range (np.ndarray): List of available keys.

    Returns:
      int: Index of the next accessed key.

    Raises:
        RuntimeError: If generating the cycle pattern fails:
            * Accessing the keys range due to invalid or empty array
              (IndexError, ValueError).
            * Using invalid argument types or non-numeric inputs (TypeError).
            * Accessing an invalid index in the cycle subset (IndexError).
    """
    try:
        # Calculate the cycle length dynamically,
        # based on number of requests generated so far
        # (cycle base ensures minimum cycle length)
        cycle_length = (
            cycle_base + (requests_count // cycle_divisor) % cycle_mod
        )

        # Select a subset of keys of
        # cycle length size, starting from
        # the head of the keys list
        cycle = keys_range[:cycle_length]

        # Determine the requested key
        # within the subset of key just
        # selected, based on the number of
        # requests generated so far as well as
        # the cycle length
        requested_key = int(cycle[requests_count % cycle_length])

        return requested_key
    except (IndexError, TypeError, ValueError) as e:
        msg = "Cycle pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "cycle_base": cycle_base,
                "cycle_divisor": cycle_divisor,
                "cycle_mod": cycle_mod,
                "requests_count": requests_count,
                "keys_range_len": (
                    len(keys_range) if keys_range is not None else None
                ),
                "context": "Cycle pattern generation",
            },
        )
        raise RuntimeError(msg) from e
