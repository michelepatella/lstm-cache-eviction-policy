from typing import Dict, Tuple

from const import HIT_COUNTER_NAME, MISS_COUNTER_NAME
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def calculate_hit_miss_rate(counters: Dict[str, int]) -> Tuple[float, float]:
    """
    Calculate cache hit and miss rates as percentages.

    This function calculates the cache hit rate and miss rate
    based on the provided counters. The rates are expressed
    as percentages of the total number of cache requests.

    Args:
        counters (Dict[str, int]): Dictionary containing hit and
                                   miss counters.

    Returns:
        Tuple[float, float]: Hit rate and miss rate (in percentages).

    Raises:
        RuntimeError: If the calculation of hit and miss rates fails e.g.:
                      * Invalid counter values.
    """
    try:
        # Compute total cache accesses
        total_cache_accesses = (
            counters[HIT_COUNTER_NAME] + counters[MISS_COUNTER_NAME]
        )

        debug(f"Total cache accesses: {total_cache_accesses}")

        # Check whether there is no cache
        # access registered
        if total_cache_accesses == 0:
            return 0.0, 0.0

        # Compute hit and miss rates in %
        hit_rate = (counters[HIT_COUNTER_NAME] / total_cache_accesses) * 100
        miss_rate = (counters[MISS_COUNTER_NAME] / total_cache_accesses) * 100

        info(f"Cache hit and miss rates calculated: {hit_rate}%, {miss_rate}%")

        return hit_rate, miss_rate
    except TypeError as e:
        msg = "Failed to calculate hit and miss rates"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
