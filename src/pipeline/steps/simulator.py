import mlflow

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
from components.visualization.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from const import (
    CACHE_LSTM_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
    SIMULATIONS_METRICS_POLICY_NAME,
    SIMULATIONS_METRICS_TIMELINE_NAME,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    CACHE_FIFO_NAME,
    CACHE_LFU_NAME,
    CACHE_LRU_NAME,
    CACHE_RANDOM_NAME,
    LOGS_SIMULATIONS_PHASE,
    PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH,
    PLOT_STATIC_HIT_MISS_RATES_FILE_PATH,
    RESULTS_DYNAMIC_SIMULATIONS_FILE_PATH,
    RESULTS_STATIC_SIMULATIONS_FILE_PATH,
    SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME,
    SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME,
    SIMULATIONS_METRICS_HIT_RATE_NAME,
    SIMULATIONS_METRICS_MISS_RATE_NAME,
)


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
    # Set the new pipeline step
    logs_phase.set(LOGS_SIMULATIONS_PHASE)
    mlflow.start_run(run_name=LOGS_SIMULATIONS_PHASE, nested=True)

    # Setup
    config = prepare_config()
    initialize_logs()

    info("Simulations started")

    # Prepare configuration
    data_distribution_mode = config.data.mode
    mistake_window = config.simulations.metrics.mistake_rate.window

    # Define cache eviction policies to simulate
    cache_eviction_policies = {
        CACHE_LRU_NAME: CacheWrapper(
            LRUCache,
            CacheMetricsLogger(),
            config,
        ),
        CACHE_LFU_NAME: CacheWrapper(
            LFUCache,
            CacheMetricsLogger(),
            config,
        ),
        CACHE_FIFO_NAME: CacheWrapper(
            FIFOCache,
            CacheMetricsLogger(),
            config,
        ),
        CACHE_RANDOM_NAME: RandomCache(
            None,
            CacheMetricsLogger(),
            config,
        ),
        CACHE_LSTM_NAME: LSTMCache(
            None,
            CacheMetricsLogger(),
            config,
        ),
    }

    # For each cache eviction policy run a simulation
    results = []
    for policy, cache in cache_eviction_policies.items():
        mlflow.start_run(
            run_name=f"{LOGS_SIMULATIONS_PHASE} — {policy}", nested=True
        )

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
            SIMULATIONS_METRICS_POLICY_NAME: policy,
            SIMULATIONS_METRICS_HIT_RATE_NAME: hit_rate,
            SIMULATIONS_METRICS_MISS_RATE_NAME: miss_rate,
            SIMULATIONS_METRICS_HIT_COUNTER_NAME: counters[
                SIMULATIONS_METRICS_HIT_COUNTER_NAME
            ],
            SIMULATIONS_METRICS_MISS_COUNTER_NAME: counters[
                SIMULATIONS_METRICS_MISS_COUNTER_NAME
            ],
            SIMULATIONS_METRICS_TIMELINE_NAME: timeline,
            SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME: eviction_mistake_rate,
            SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME: avg_cache_latency,
        }

        # Save metrics
        results.append(metrics)

        mlflow.log_metrics(
            {
                "num_hits": counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME],
                "num_misses": counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "eviction_mistake_rate": eviction_mistake_rate,
                "min_cache_latency": min(cache_latencies),
                "max_cache_latency": max(cache_latencies),
                "avg_cache_latency": avg_cache_latency,
            }
        )
        mlflow.end_run()

    # Determine results and plot file path according
    # to data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        results_file_path = RESULTS_STATIC_SIMULATIONS_FILE_PATH
        plot_save_path = PLOT_STATIC_HIT_MISS_RATES_FILE_PATH
    else:
        results_file_path = RESULTS_DYNAMIC_SIMULATIONS_FILE_PATH
        plot_save_path = PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH

    # Save simulations results
    save_simulations_metrics(results, results_file_path)

    # Plot hit and miss rates over time
    plot_hit_miss_rate(
        [
            {
                SIMULATIONS_METRICS_POLICY_NAME: r[
                    SIMULATIONS_METRICS_POLICY_NAME
                ],
                SIMULATIONS_METRICS_TIMELINE_NAME: r[
                    SIMULATIONS_METRICS_TIMELINE_NAME
                ],
            }
            for r in results
        ],
        plot_save_path,
    )

    mlflow.log_param("cache_policies", list(cache_eviction_policies.keys()))
    mlflow.log_artifacts(results_file_path)
    mlflow.log_artifacts(plot_save_path)
    mlflow.end_run()

    info("Simulations completed")
