from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.evaluation.simulations.metrics.calculations.eviction_mistake_rate_calculator import (
    calculate_eviction_mistake_rate,
)
from components.evaluation.simulations.metrics.calculations.hit_miss_rates_calculator import (
    calculate_hit_miss_rate,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average
from src.const import (
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
)


def calculate_simulation_metrics(
    counters: dict[str, int],
    cache_latencies: list[float],
    mistake_window: int,
    metrics_logger: CacheMetricsLogger,
    hit_counter_name: str = SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    miss_counter_name: str = SIMULATIONS_METRICS_MISS_COUNTER_NAME,
) -> tuple[float, float, float, float]:
    """Calculate cache simulations metrics.

    This function calculates the main performance metrics for a cache
    simulation, including hit rate, miss rate, eviction mistake rate,
    and average cache latency. It uses the provided counters and
    cache latency records, as well as a metrics logger for eviction tracking.

    Args:
        counters (dict[str, int]): Dictionary containing counts of cache
                                   hits and misses.
        cache_latencies (list[float]): List of cache access latencies
                                       recorded during simulation.
        mistake_window (int): Mistake window for mistake rate calculation.
        metrics_logger (CacheMetricsLogger): Object logging cache events.
        hit_counter_name (str): Name of the hit counter in the counters'
                                dictionary.
        miss_counter_name (str): Name of the miss counter in the counters'
                                 dictionary.

    Returns:
        tuple[float, float, float, float]:
            - hit_rate: Cache hit rate as a percentage.
            - miss_rate: Cache miss rate as a percentage.
            - eviction_mistake_rate: Rate of eviction mistakes over
                                     the given window.
            - avg_cache_latency: Average latency of cache accesses.

    Raises:
        RuntimeError: If simulation metrics calculation fails:
            * Hit and miss counters not found in counters dictionary
              (KeyError).
            * Metrics logger missing required attributes (AttributeError).
            * Invalid data types in counters or latencies (TypeError).
    """
    try:
        debug(
            "Simulation metrics calculation started",
            extra={
                "hit_counter_name": hit_counter_name,
                "miss_counter_name": miss_counter_name,
                "hit_count": counters.get(hit_counter_name),
                "miss_count": counters.get(miss_counter_name),
                "cache_latencies_num": len(cache_latencies),
                "mistake_window": mistake_window,
                "context": "Simulation metrics calculation",
            },
        )

        # Get total cache accesses
        total_cache_accesses = (
            counters[hit_counter_name] + counters[miss_counter_name]
        )

        # Calculate hit and miss rates
        hit_rate, miss_rate = calculate_hit_miss_rate(
            counters[hit_counter_name],
            counters[miss_counter_name],
            total_cache_accesses,
        )

        # Extract eviction and access data
        evicted_items = metrics_logger.evicted_keys
        access_events_dict = metrics_logger.access_events

        # Calculate eviction mistake rate
        eviction_mistake_rate = calculate_eviction_mistake_rate(
            evicted_items,
            access_events_dict,
            mistake_window,
        )

        # Calculate average cache latency
        avg_cache_latency = calculate_average(cache_latencies)
        info(
            "Average cache latency calculated",
            extra={
                "cache_latency_avg": avg_cache_latency,
                "latencies_num": len(cache_latencies),
                "context": "Simulation metrics calculation",
            },
        )

        debug(
            "Simulation metrics calculation completed",
            extra={
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "eviction_mistake_rate": eviction_mistake_rate,
                "cache_latency_avg": avg_cache_latency,
                "cache_accesses_num": total_cache_accesses,
                "evicted_items_num": len(evicted_items),
                "access_events_num": len(access_events_dict),
                "context": "Simulation metrics calculation",
            },
        )

        return (
            hit_rate,
            miss_rate,
            eviction_mistake_rate,
            avg_cache_latency,
        )
    except (KeyError, AttributeError, TypeError) as e:
        msg = "Simulation metrics calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "hit_counter_name": hit_counter_name,
                "miss_counter_name": miss_counter_name,
                "counters_keys": list(counters.keys()),
                "cache_latencies_num": len(cache_latencies),
                "mistake_window": mistake_window,
                "context": "Simulation metrics calculation",
            },
        )
        raise RuntimeError(msg) from e
