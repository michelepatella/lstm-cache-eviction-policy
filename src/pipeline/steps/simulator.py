from components.caches.implementations.fifo_cache import FIFOCache
from components.caches.implementations.lfu_cache import LFUCache
from components.caches.implementations.lru_cache import LRUCache
from components.caches.implementations.lstm_cache import LSTMCache
from components.caches.implementations.random_cache import RandomCache
from components.caches.simulations.runner import (
    run_cache_simulation,
)
from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.caches.utils.cache_wrapper import (
    CacheWrapper,
)
from components.evaluation.simulations.metrics.calculator import (
    calculate_simulation_metrics,
)
from components.evaluation.simulations.metrics.io.saver import (
    save_simulations_metrics,
)
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.visualization.plots.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from components.visualization.reports.simulation_reporter import (
    generate_simulations_report,
)
from const import (
    AVG_CACHE_LATENCY_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    DYNAMIC_SIMULATIONS_RESULTS_FILE_NAME,
    EVICTION_MISTAKE_RATE_NAME,
    FIFO_CACHE_NAME,
    HIT_COUNTER_NAME,
    HIT_MISS_RATES_PLOT_FILE_NAME,
    HIT_RATE_NAME,
    LFU_CACHE_NAME,
    LOGS_SIMULATIONS_PHASE,
    LRU_CACHE_NAME,
    LSTM_CACHE_NAME,
    MISS_COUNTER_NAME,
    MISS_RATE_NAME,
    PLOTS_DIRECTORY_PATH,
    POLICY_NAME,
    RANDOM_CACHE_NAME,
    RESULTS_DIRECTORY_PATH,
    STATIC_SIMULATIONS_RESULTS_FILE_NAME,
    TIMELINE_NAME,
)
from pipeline.config.configurator import prepare_config


def run_simulations() -> None:
    """
    Run cache simulations for multiple cache eviction policies.

    This function executes a complete cache simulations workflow
    across different cache eviction strategies. For each policy,
    it initializes the cache, runs the cache simulation, calculates
    key performance metrics, saves the results, and plots performance data.

    Returns:
        None
    """
    # Set new state
    logs_phase.set(LOGS_SIMULATIONS_PHASE)

    # Setup
    config = prepare_config()
    initialize_logs()

    # Prepare configuration
    data_distribution_mode = config.data.mode
    mistake_window = config.simulations.metrics.mistake_rate.window

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
        LSTM_CACHE_NAME: LSTMCache(
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
        ) = calculate_simulation_metrics(
            counters, cache_latencies, mistake_window, cache.metrics_logger
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

    # Generate a report for simulations results
    generate_simulations_report(results)

    # Determine results file name according
    # to data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        results_file_name = STATIC_SIMULATIONS_RESULTS_FILE_NAME
    else:
        results_file_name = DYNAMIC_SIMULATIONS_RESULTS_FILE_NAME

    # Build results save path
    results_save_path = (
        RESULTS_DIRECTORY_PATH / data_distribution_mode / results_file_name
    )

    # Save simulations results
    save_simulations_metrics(results, results_save_path)

    # Plot hit and miss rates over time
    hit_miss_rate_plot_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / HIT_MISS_RATES_PLOT_FILE_NAME
    )
    plot_hit_miss_rate(results, hit_miss_rate_plot_save_path)

    info("Cache simulations completed")
