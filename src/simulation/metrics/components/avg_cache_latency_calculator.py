from typing import List

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def calculate_avg_cache_latency(
    cache_latencies: List[float],
) -> float:
    """
    Calculate the average cache latency.

    This function computes the average latency of cache,
    given a list of recorded latency values. If the list
    is empty, the result is set to None.

    Args:
        cache_latencies (List[float]): List of cache latencies.

    Returns:
        float: Average cache latency, or None
               if no latency values are provided.
    """
    avg_cache_latency = None

    debug(f"Number of cache latencies to average: {len(cache_latencies)}")

    if len(cache_latencies) != 0:
        # Calculate average cache latency
        avg_cache_latency = sum(cache_latencies) / len(cache_latencies)

    info(f"Cache average latency calculated: {avg_cache_latency}")

    return avg_cache_latency
