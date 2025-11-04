"""hit_miss_rates_calculator.py

Utility module for calculating cache hit and miss rates.

This module provides the `calculate_hit_miss_rate` function, which
computes the hit and miss rates of a cache given the number of hits,
number of misses, and total cache accesses. The rates are expressed as
percentages.

Functions:
    calculate_hit_miss_rate(
        num_hits: int,
        num_misses: int,
        total_cache_accesses: int
    ) -> tuple[float, float]
        Returns the hit rate and miss rate as percentages.
"""

from components.logs.levels.info_logger import info
from components.math.percentage_calculator import calculate_percentage


def calculate_hit_miss_rate(
    num_hits: int,
    num_misses: int,
    total_cache_accesses: int,
) -> tuple[float, float]:
    """Calculate hit and miss rates.

    This function calculates the hit and miss rates,
    provided their counters along with total cache accesses.

    Args:
        num_hits (int): The number of hits.
        num_misses (int): The number of misses.
        total_cache_accesses (int): The total number of cache accesses.

    Returns:
        tuple[float, float]: The hit and miss rates calculated.
    """
    # Calculate hit and miss rates
    hit_rate = calculate_percentage(num_hits, total_cache_accesses)
    miss_rate = calculate_percentage(num_misses, total_cache_accesses)

    info(
        "Hit/miss rates calculated",
        extra={
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "hits_num": num_hits,
            "misses_num": num_misses,
            "cache_accesses_num": total_cache_accesses,
            "context": "Hit/miss rates calculation",
        },
    )

    return hit_rate, miss_rate
