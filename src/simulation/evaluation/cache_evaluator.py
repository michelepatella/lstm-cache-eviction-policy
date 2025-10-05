from simulation.evaluation.metrics.eviction_mistake_rate_calculator import (
    calculate_eviction_mistake_rate,
)
from simulation.evaluation.metrics.hit_miss_rate_calculator import (
    calculate_hit_miss_rate,
)
from simulation.evaluation.metrics.avg_cache_latency_calculator import (
    calculate_avg_cache_latency,
)
from utils.logs.levels.info_logger import info


def evaluate_cache(
    counters,
    tot_prefetch,
    autoregressive_latencies,
    metrics_logger,
):
    """
    Method to orchestrate cache metrics calculation.
    :param counters: A counter used while simulating a cache policy.
    :param tot_prefetch: The total number of prefetches.
    :param autoregressive_latencies: The autoregressive latencies.
    :param metrics_logger: The metrics logger.
    :return: All the computed cache metrics.
    """
    # initial message
    info("🔄 Cache metrics calculation started...")

    # calculate hit rate and miss rate
    (hit_rate, miss_rate) = calculate_hit_miss_rate(counters)

    # component evaluation
    eviction_mistake_rate = calculate_eviction_mistake_rate(metrics_logger)

    # calculate avg prefetching average
    avg_prefetching_latency = calculate_avg_cache_latency(
        autoregressive_latencies
    )

    # show a successful message
    info("🟢 Cache metrics calculated.")

    return (
        hit_rate,
        miss_rate,
        eviction_mistake_rate,
        avg_prefetching_latency,
    )
