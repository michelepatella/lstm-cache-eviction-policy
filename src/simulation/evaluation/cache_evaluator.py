from typing import Dict, List, Tuple

from config.classes.Config import Config
from simulation.evaluation.metrics.eviction_mistake_rate_calculator import (
    calculate_eviction_mistake_rate,
)
from simulation.evaluation.metrics.hit_miss_rate_calculator import (
    calculate_hit_miss_rate,
)
from simulation.evaluation.metrics.avg_cache_latency_calculator import (
    calculate_avg_cache_latency,
)
from simulation.utils.CacheMetricsLogger import CacheMetricsLogger
from utils.logs.levels.info_logger import info


def evaluate_cache(
    counters: Dict[str, int],
    cache_latencies: List[float],
    metrics_logger: CacheMetricsLogger,
    config: Config,
) -> Tuple[float, float, float, float]:
    """
    Evaluate cache performance metrics.

    This function calculates the main performance metrics for a cache
    simulation, including hit rate, miss rate, eviction mistake rate,
    and average cache latency. It uses the provided counters and
    cache latency records, as well as a metrics logger for eviction
    tracking.

    Parameters:
        counters (Dict[str, int]): Dictionary containing counts of cache
                                   hits and misses.
        cache_latencies (List[float]): List of cache access latencies
                                       recorded during simulation.
        metrics_logger (CacheMetricsLogger): Object logging cache
                                             events.
        config (Config): Configuration object.

    Returns:
        Tuple[float, float, float, float]: A tuple containing cache hit rate,
                                           miss rate, eviction mistake, and
                                           average latency.
    """
    # Prepare configuration
    mistake_window = config.simulation.evaluation.mistake_rate.window

    # Calculate hit and miss rates
    hit_rate, miss_rate = calculate_hit_miss_rate(counters)

    # Calculate eviction mistake rate
    eviction_mistake_rate = calculate_eviction_mistake_rate(
        metrics_logger, mistake_window
    )

    # Calculate average cache latency
    avg_cache_latency = calculate_avg_cache_latency(cache_latencies)

    info("Cache metrics calculated")

    return (
        hit_rate,
        miss_rate,
        eviction_mistake_rate,
        avg_cache_latency,
    )
