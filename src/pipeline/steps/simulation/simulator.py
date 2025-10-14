from pipeline.const import (
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
    TIMELINE_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    STATIC_SIMULATION_RESULTS_FILE_NAME,
    DYNAMIC_SIMULATION_RESULTS_FILE_NAME,
)
from pipeline.config.configurator import prepare_config
from pipeline.steps.simulation.caches.FIFOCache import FIFOCache
from pipeline.steps.simulation.caches.LFUCache import LFUCache
from pipeline.steps.simulation.caches.LRUCache import LRUCache
from pipeline.steps.simulation.caches.LSTMCache import LSTMCache
from pipeline.steps.simulation.caches.RandomCache import RandomCache
from pipeline.steps.simulation.caches.utils.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from pipeline.steps.simulation.caches.utils.classes.CacheWrapper import (
    CacheWrapper,
)
from pipeline.steps.simulation.metrics.calculator import (
    calculate_cache_simulation_metrics,
)
from pipeline.steps.simulation.metrics.io.saver import save_simulation_results
from pipeline.steps.simulation.running.simulation_runner import (
    run_cache_simulation,
)
from pipeline.steps.simulation.visualization.plotting.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from pipeline.steps.simulation.visualization.reporting.simulation_reporter import (
    generate_simulation_report,
)
from utils.json.saver import save_json
from utils.logs.initializer import logs_phase
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def run_simulations() -> None:
    """
    Run cache simulations for multiple cache eviction policies.

    This function executes a complete cache simulation workflow
    across different cache eviction strategies. For each policy,
    it initializes the cache, runs the cache simulation, calculates
    key performance metrics, saves the results, and plots performance data.

    Returns:
        None
    """
    # Set new state
    logs_phase.set(LOGS_SIMULATION_PHASE)

    # Read configuration
    config = prepare_config()

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode
    mistake_window = config.simulation.metrics.mistake_rate.window

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
        ) = calculate_cache_simulation_metrics(
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

    # Generate a report for simulation results
    generate_simulation_report(results)

    # Determine results file name according
    # to data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        results_file_name = STATIC_SIMULATION_RESULTS_FILE_NAME
    else:
        results_file_name = DYNAMIC_SIMULATION_RESULTS_FILE_NAME

    # Build results save path
    results_save_path = (
        RESULTS_DIRECTORY_PATH / data_distribution_mode / results_file_name
    )

    # Save simulation results
    save_simulation_results(results, results_save_path)

    # Plot hit and miss rates over time
    hit_miss_rate_plot_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / HIT_MISS_RATES_PLOT_FILE_NAME
    )
    plot_hit_miss_rate(results, hit_miss_rate_plot_save_path)

    info("Cache simulations completed")
