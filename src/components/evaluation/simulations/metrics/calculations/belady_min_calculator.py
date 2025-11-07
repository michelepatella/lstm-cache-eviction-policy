"""belady_min_calculator.py

Utility module for calculating Belady's MIN cache performance metrics.

This module provides the `calculate_belady_min` function, which computes
cache hit and miss rates for a given access sequence using Belady's optimal
(MIN) replacement policy. The algorithm evicts the key that will be accessed
farthest in the future (or never again) when the cache is full.

Functions:
    calculate_belady_min(
        access_sequence: list[int],
        cache_size: int
    ) -> tuple[float, float]
        Calculates the hit and miss rates of Belady's MIN policy for the
        provided access sequence and cache size.
"""

from collections import defaultdict, deque

import numpy as np

from components.const import LIST_FIRST_IDX
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.math.percentage_calculator import calculate_percentage


def calculate_belady_min(
    access_sequence: list[int],
    cache_size: int,
) -> tuple[float, float]:
    """Calculate Belady's MIN cache hit and miss rates.

    This function calculates Belady's MIN cache hit and
    miss rates for a given access sequence. Belady's MIN algorithm
    evicts the key that will be accessed farthest in the future
    (or never accessed again) when the cache is full.

    Args:
        access_sequence (list[int]): Sequence of accessed keys.
        cache_size (int): Size of the cache.

    Returns:
        tuple[float, float]:
            - hit_rate: Percentage of accesses that hit in the cache.
            - miss_rate: Percentage of accesses that miss in the cache.

    Raises:
        RuntimeError: If Belady MIN calculation fails:
            * Invalid values in access sequence or cache size
              (ValueError).
            * Wrong type for access sequence or cache size (TypeError).
            * Unexpected empty deque while processing future key
              positions (IndexError).
    """
    try:
        info(
            "Belady MIN calculation started",
            extra={
                "access_sequence_len": len(access_sequence),
                "cache_size": cache_size,
                "context": "Belady MIN",
            },
        )

        # Initialization
        cache = set()
        next_use_positions = defaultdict(deque)
        hits_num = 0
        misses_num = 0

        # Pre-calculate future positions of each key
        for idx, key in enumerate(access_sequence):
            next_use_positions[key].append(idx)

        # For each key access in the sequence
        for _, key in enumerate(access_sequence):
            # Remove the current key request
            # as already served
            next_use_positions[key].popleft()

            # Check whether the requested
            # key is in cache or not, keeping
            # track of the number of hits and
            # misses consequently
            if key in cache:
                hits_num += 1
            else:
                misses_num += 1

                # Check whether the cache is
                # full and we need to evict an item
                if len(cache) < cache_size:
                    cache.add(key)
                else:
                    # Remove the key intended
                    # to be accessed later than others
                    # (or never more)
                    farthest_next_use = -1
                    key_to_evict = None
                    for k in cache:
                        if next_use_positions[k]:
                            next_use = next_use_positions[k][LIST_FIRST_IDX]
                        else:
                            next_use = np.inf
                        if next_use > farthest_next_use:
                            farthest_next_use = next_use
                            key_to_evict = k

                    # Remove the identified key,
                    # inserting the requested one
                    cache.remove(key_to_evict)
                    cache.add(key)

        # Calculate both hit and miss rates
        total_accesses = len(access_sequence)
        hit_rate = calculate_percentage(hits_num, total_accesses)
        miss_rate = calculate_percentage(misses_num, total_accesses)

        info(
            "Belady MIN calculation completed",
            extra={
                "hits_num": hits_num,
                "misses_num": misses_num,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "context": "Belady MIN",
            },
        )

        return hit_rate, miss_rate
    except (ValueError, TypeError, IndexError) as e:
        msg = "Belady MIN calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "access_sequence_len": len(access_sequence)
                if access_sequence
                else 0,
                "cache_size": cache_size,
                "context": "Belady MIN calculation",
            },
        )
        raise RuntimeError(msg) from e
