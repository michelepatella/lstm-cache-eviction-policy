from typing import Tuple

from components.logs.levels.info_logger import info
from components.math.percentage_calculator import calculate_percentage


def calculate_hit_miss_rate(
    num_hits: int, num_misses: int, total_cache_accesses: int
) -> Tuple[float, float]:
    """
    Calculate hit and miss rates.

    This function calculates the hit and miss rates,
    provided their counters along with total cache accesses.

    Args:
        num_hits (int): The number of hits.
        num_misses (int): The number of misses.
        total_cache_accesses (int): The total number of cache accesses.

    Returns:
        Tuple[float, float]: The hit and miss rates calculated.
    """
    # Calculate hit and miss rates
    hit_rate = calculate_percentage(num_hits, total_cache_accesses)
    miss_rate = calculate_percentage(num_misses, total_cache_accesses)

    info(f"Hit and miss rates: {hit_rate}%, {miss_rate}%")

    return hit_rate, miss_rate
