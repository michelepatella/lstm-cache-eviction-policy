from typing import Dict, List, Tuple

from components.evaluation.simulations.metrics.calculations.eviction_mistake_rate_calculator import \
    calculate_eviction_mistake_rate
from components.evaluation.simulations.metrics.calculations.hit_miss_rates_calculator import (
    calculate_hit_miss_rate,
)
from const import HIT_COUNTER_NAME, MISS_COUNTER_NAME
from components.caches.utils.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average


def calculate_simulation_metrics(
    counters: Dict[str, int],
    cache_latencies: List[float],
    mistake_window: int,
    metrics_logger: CacheMetricsLogger,
) -> Tuple[float, float, float, float]:
    """
    Calculate cache simulations metrics.

    This function calculates the main performance metrics for a cache
    simulation, including hit rate, miss rate, eviction mistake rate,
    and average cache latency. It uses the provided counters and
    cache latency records, as well as a metrics logger for eviction
    tracking.

    Args:
        counters (Dict[str, int]): Dictionary containing counts of cache
                                   hits and misses.
        cache_latencies (List[float]): List of cache access latencies
                                       recorded during simulation.
        mistake_window (int): Mistake window for mistake rate calculation.
        metrics_logger (CacheMetricsLogger): Object logging cache
                                             events.

    Returns:
        Tuple[float, float, float, float]:
            - hit_rate: Cache hit rate as a percentage.
            - miss_rate: Cache miss rate as a percentage.
            - eviction_mistake_rate: Rate of eviction mistakes over the given window.
            - avg_cache_latency: Average latency of cache accesses.
    """
    # Get total cache accesses
    total_cache_accesses = (
        counters[HIT_COUNTER_NAME] + counters[MISS_COUNTER_NAME]
    )

    # Calculate hit and miss rates
    hit_rate, miss_rate = calculate_hit_miss_rate(
        counters[HIT_COUNTER_NAME],
        counters[MISS_COUNTER_NAME],
        total_cache_accesses,
    )

    # Extract eviction and access data
    evicted_items = metrics_logger.evicted_keys
    access_events_dict = metrics_logger.access_events

    # Calculate eviction mistake rate
    eviction_mistake_rate = calculate_eviction_mistake_rate(
        evicted_items, access_events_dict, mistake_window
    )

    # Calculate average cache latency
    avg_cache_latency = calculate_average(cache_latencies)

    info("Cache simulation metrics calculated")

    return (
        hit_rate,
        miss_rate,
        eviction_mistake_rate,
        avg_cache_latency,
    )
