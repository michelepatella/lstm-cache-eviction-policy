import logging
import os

import dagshub
import numpy as np
from dotenv import load_dotenv

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
from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.rows.extractions.lasts_extractor import (
    extract_last_rows_from_dataset,
)
from components.evaluation.simulations.metrics.calculations.belady_min_calculator import (
    calculate_belady_min,
)
from components.evaluation.simulations.metrics.calculator import (
    calculate_simulation_metrics,
)
from components.evaluation.simulations.metrics.io.saver import (
    save_simulations_metrics,
)
from components.logs.handlers.elastic_handler import ElasticHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.visualization.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    CACHE_LFU_NAME,
    CACHE_LRU_NAME,
    CACHE_RANDOM_NAME,
    DAGS_HUB_ENV_VAR_REPO_NAME,
    DAGS_HUB_ENV_VAR_REPO_OWNER_NAME,
    DAGS_HUB_MLFLOW_ENABLED,
    LOGS_PHASE_SIMULATIONS,
    PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH,
    PLOT_STATIC_HIT_MISS_RATES_FILE_PATH,
    RESULTS_DYNAMIC_SIMULATIONS_FILE_PATH,
    RESULTS_STATIC_SIMULATIONS_FILE_PATH,
    SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME,
    SIMULATIONS_METRICS_BELADY_MIN_HIT_RATE_NAME,
    SIMULATIONS_METRICS_BELADY_MIN_MISS_RATE_NAME,
    SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME,
    SIMULATIONS_METRICS_HIT_RATE_NAME,
    SIMULATIONS_METRICS_MISS_RATE_NAME,
)
from src.const import (
    CACHE_LSTM_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_TESTING_SPLIT_TYPE,
    MLFLOW_NESTED_ENABLED,
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
    SIMULATIONS_METRICS_POLICY_NAME,
    SIMULATIONS_METRICS_TIMELINE_NAME,
)

# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_ENV_VAR_REPO_OWNER_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_ENV_VAR_REPO_NAME)


def run_simulations() -> None:
    """Run cache simulations for multiple cache eviction policies.

    This function executes a complete cache simulations workflow
    across different cache eviction strategies. For each policy,
    it initializes the cache, runs the cache simulation, calculates
    key performance metrics, saves the results, and plots performance data.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_SIMULATIONS)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_PHASE_SIMULATIONS,
        nested=MLFLOW_NESTED_ENABLED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs()

        # Prepare configuration
        data_distribution_mode = config.data.mode
        mistake_window = config.simulations.metrics.mistake_rate.window
        testing_batch_size = config.testing.general.batch_size
        testing_shuffle = config.testing.general.shuffle
        cache_size = config.simulations.cache.dimension

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

        # Get testing set
        testing_set, testing_loader = initialize_data_loader(
            DATASET_TESTING_SPLIT_TYPE,
            testing_batch_size,
            testing_shuffle,
            AccessLogsDataset,
            config,
        )

        info(
            "Simulations started",
            extra={
                "data_distribution_mode": data_distribution_mode,
                "policies_simulated": list(cache_eviction_policies.keys()),
                "mistake_window": mistake_window,
                "context": "Simulations",
            },
        )

        # For each cache eviction policy run a simulation
        results = []
        for policy, cache in cache_eviction_policies.items():
            with mlflow.start_run(
                run_name=f"{LOGS_PHASE_SIMULATIONS} ({policy})",
                nested=MLFLOW_NESTED_ENABLED,
            ):
                # Simulate a cache policy and
                # get simulation insights
                counters, timeline, cache_latencies = run_cache_simulation(
                    cache,
                    policy,
                    testing_set,
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
                    counters,
                    cache_latencies,
                    mistake_window,
                    cache.metrics_logger,
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

                # Experiment tracking
                mlflow.log_metrics(
                    {
                        "requests_num": counters[
                            SIMULATIONS_METRICS_HIT_COUNTER_NAME
                        ]
                        + counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                        "hits_num": counters[
                            SIMULATIONS_METRICS_HIT_COUNTER_NAME
                        ],
                        "misses_num": counters[
                            SIMULATIONS_METRICS_MISS_COUNTER_NAME
                        ],
                        "hit_rate": hit_rate,
                        "miss_rate": miss_rate,
                        "eviction_mistake_rate": eviction_mistake_rate,
                        "latency_min": min(cache_latencies),
                        "latency_max": max(cache_latencies),
                        "latency_avg": avg_cache_latency,
                        "latency_std": np.std(cache_latencies),
                    },
                )

        # Extract key access sequence
        # to pass to Belady MIN benchmark
        testing_rows = extract_last_rows_from_dataset(
            len(testing_set) - 1,
            len(testing_set),
            testing_set.data,
        )
        access_sequence = [key for _, key in testing_rows]

        # Calculate Belady MIN (benchmark) and save them
        # into results
        belady_min_hit_rate, belady_min_miss_rate = calculate_belady_min(
            access_sequence,
            cache_size,
        )
        results.append(
            {
                SIMULATIONS_METRICS_BELADY_MIN_HIT_RATE_NAME: belady_min_hit_rate,
                SIMULATIONS_METRICS_BELADY_MIN_MISS_RATE_NAME: belady_min_miss_rate,
            },
        )

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
                if SIMULATIONS_METRICS_POLICY_NAME in r
            ],
            plot_save_path,
        )

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_param(
            "api_kwargs",
            cache_eviction_policies[CACHE_LSTM_NAME]._api_kwargs,
        )
        mlflow.log_artifact(results_file_path)
        mlflow.log_artifact(plot_save_path)

    info(
        "Simulations completed",
        extra={
            "results": [
                {
                    "policy": r[SIMULATIONS_METRICS_POLICY_NAME],
                    "hit_rate": r[SIMULATIONS_METRICS_HIT_RATE_NAME],
                    "miss_rate": r[SIMULATIONS_METRICS_MISS_RATE_NAME],
                    "eviction_mistake_rate": r[
                        SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME
                    ],
                    "cache_latency_avg": r[
                        SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME
                    ],
                    "hits_num": r[SIMULATIONS_METRICS_HIT_COUNTER_NAME],
                    "misses_num": r[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                }
                for r in results
                if SIMULATIONS_METRICS_POLICY_NAME in r
            ],
            "results_save_path": str(results_file_path),
            "plot_save_paths": str(plot_save_path),
            "context": "Simulations",
        },
    )


if __name__ == "__main__":
    run_simulations()

    # Force logs flush
    for handler in logging.getLogger().handlers:
        if isinstance(handler, ElasticHandler):
            handler.flush_buffer()
