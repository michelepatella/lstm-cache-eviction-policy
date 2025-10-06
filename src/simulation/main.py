from config.classes.Config import Config
from lstm_cache_eviction_policy.LSTMCache import LSTMCache
from simulation.baseline_caches.FIFOCache import FIFOCache
from simulation.baseline_caches.LFUCache import LFUCache
from simulation.baseline_caches.LRUCache import LRUCache
from simulation.baseline_caches.RandomCache import RandomCache
from simulation.baseline_caches.utils.CacheWrapper import CacheWrapper
from simulation.metrics.calculator import calculate_cache_simulation_metrics
from utils.simulation.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from simulation.visualization.plotting.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from simulation.visualization.reporting.simulation_reporter import (
    generate_simulation_report,
)
from simulation.running.simulation_runner import run_cache_simulation
from utils.logs.levels.info_logger import info


def run_simulations(config: Config) -> None:
    # Prepare configuration
    data_distribution_mode = config.data.general.mode

    # Data setup and initialization
    cache_eviction_policies = {
        "LRU": CacheWrapper(
            LRUCache,
            CacheMetricsLogger(),
            config,
        ),
        "LFU": CacheWrapper(
            LFUCache,
            CacheMetricsLogger(),
            config,
        ),
        "FIFO": CacheWrapper(
            FIFOCache,
            CacheMetricsLogger(),
            config,
        ),
        "RANDOM": RandomCache(
            None,
            CacheMetricsLogger(),
            config,
        ),
        "LSTM": LSTMCache(
            None,
            CacheMetricsLogger(),
            config,
        ),
    }
    results = []

    # For each cache eviction policy run a simulation
    for policy, cache in cache_eviction_policies.items():
        # Simulate a cache policy and
        # get some numbers to calculate metrics
        counters, timeline, cache_latencies = run_cache_simulation(
            cache,
            policy,
            config,
        )

        # Calculate metrics at the end
        # of cache simulation
        (
            hit_rate,
            miss_rate,
            eviction_mistake_rate,
            avg_cache_latency,
        ) = calculate_cache_simulation_metrics(
            counters, cache_latencies, cache.metrics_logger, config
        )

        return {
            "policy": policy,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "hits": counters["hits"],
            "misses": counters["misses"],
            "avg_prefetching_latency": avg_cache_latency,
            "timeline": timeline,
            "eviction_mistake_rate": eviction_mistake_rate,
        }

        results.append(result)

    # Generate a report for simulation results
    generate_simulation_report(results)

    # Plot hit and miss rates over time
    plot_hit_miss_rate(results, data_distribution_mode)

    # print a successful message
    info("✅ Cache simulations completed.")
