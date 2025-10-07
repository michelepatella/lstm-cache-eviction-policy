from config.classes.Config import Config
from const import (
    AVG_CACHE_LATENCY_NAME,
    EVICTION_MISTAKE_RATE_NAME,
    FIFO_CACHE_NAME,
    HIT_COUNTER_NAME,
    HIT_MISS_RATES_PLOT_FILE_NAME,
    HIT_RATE_NAME,
    LFU_CACHE_NAME,
    LOGS_SIMULATION_PHASE,
    LRU_CACHE_NAME,
    LSTM_CACHE_NAME,
    MISS_COUNTER_NAME,
    MISS_RATE_NAME,
    PLOTS_DIRECTORY_PATH,
    POLICY_NAME,
    RANDOM_CACHE_NAME,
    RESULTS_DIRECTORY_PATH,
    SIMULATION_RESULTS_FILE_NAME,
    TIMELINE_NAME,
)
from lstm_cache_eviction_policy.LSTMCache import LSTMCache
from simulation.baseline_caches.FIFOCache import FIFOCache
from simulation.baseline_caches.LFUCache import LFUCache
from simulation.baseline_caches.LRUCache import LRUCache
from simulation.baseline_caches.RandomCache import RandomCache
from simulation.baseline_caches.utils.CacheWrapper import CacheWrapper
from simulation.metrics.calculator import calculate_cache_simulation_metrics
from simulation.running.simulation_runner import run_cache_simulation
from simulation.visualization.plotting.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from simulation.visualization.reporting.simulation_reporter import (
    generate_simulation_report,
)
from utils.json.saver import save_json
from utils.logs.initializer import logs_phase
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from utils.simulation.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)


def run_simulations(config: Config) -> None:
    """
    Run cache simulations for multiple cache eviction policies.

    This function executes a complete cache simulation workflow
    across different cache eviction strategies. For each policy,
    it initializes the cache, runs the cache simulation, calculates
    key performance metrics, saves the results, and plots performance data.

    Parameters:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set new state
    logs_phase.set(LOGS_SIMULATION_PHASE)

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode

    # Data setup and initialization
    cache_eviction_policies = {
        LRU_CACHE_NAME: CacheWrapper(
            LRUCache,
            CacheMetricsLogger(),
            config,
        ),
        LFU_CACHE_NAME: CacheWrapper(
            LFUCache,
            CacheMetricsLogger(),
            config,
        ),
        FIFO_CACHE_NAME: CacheWrapper(
            FIFOCache,
            CacheMetricsLogger(),
            config,
        ),
        RANDOM_CACHE_NAME: RandomCache(
            None,
            CacheMetricsLogger(),
            config,
        ),
    }
    results = []

    # For each cache eviction policy run a simulation
    for policy, cache in cache_eviction_policies.items():
        # Simulate a cache policy and
        # get simulation insights
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

        # Collect metrics together for the
        # current cache eviction policy
        metrics = {
            POLICY_NAME: policy,
            HIT_RATE_NAME: hit_rate,
            MISS_RATE_NAME: miss_rate,
            HIT_COUNTER_NAME: counters[HIT_COUNTER_NAME],
            MISS_COUNTER_NAME: counters[MISS_COUNTER_NAME],
            TIMELINE_NAME: timeline,
            EVICTION_MISTAKE_RATE_NAME: eviction_mistake_rate,
            AVG_CACHE_LATENCY_NAME: avg_cache_latency,
        }

        # Save metrics
        results.append(metrics)

    # Generate a report for simulation results
    generate_simulation_report(results)

    # Save simulation results as JSON
    results_save_path = (
        RESULTS_DIRECTORY_PATH
        / data_distribution_mode
        / SIMULATION_RESULTS_FILE_NAME
    )
    save_json(results, results_save_path)

    debug("Simulation results saved to JSON file")

    # Plot hit and miss rates over time
    hit_miss_rate_plot_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / HIT_MISS_RATES_PLOT_FILE_NAME
    )
    plot_hit_miss_rate(results, hit_miss_rate_plot_save_path)

    info("Cache simulations completed")
