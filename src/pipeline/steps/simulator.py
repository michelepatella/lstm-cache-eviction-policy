"""simulator.py

Pipeline step module responsible for executing and evaluating various
cache eviction policy simulations.

This module provides the `run_simulations` function, which orchestrates
the simulation of multiple cache strategies against the test dataset.
It calculates key metrics (hit rate, miss rate, eviction mistake rate,
latency), benchmarks against Belady's optimal, saves the simulation results,
and generates performance plots.

Functions:
    run_simulations() -> None
        Runs the full cache simulation workflow for defined policies,
        saves metrics, and generates visualization plots.
"""

import logging

import dagshub
import mlflow
import numpy as np
import ray

from components.caches.implementations.lfu_cache import LFUCache
from components.caches.implementations.lru_cache import LRUCache
from components.caches.implementations.lstm_cache import LSTMCache
from components.caches.implementations.random_cache import RandomCache
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
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.ray.initializer import initialize_ray
from components.ray.tasks.caches.simulator import run_cache_simulation_task
from components.seed.setter import set_seed
from components.visualization.hit_miss_rates_plotter import (
    plot_hit_miss_rate,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import (
    CACHE_LFU_NAME,
    CACHE_LRU_NAME,
    CACHE_RANDOM_NAME,
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    DATASET_PROCESSED_TYPE,
    LOGS_PHASE_SIMULATIONS,
    PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH,
    PLOT_REAL_HIT_MISS_RATES_FILE_PATH,
    PLOT_STATIC_HIT_MISS_RATES_FILE_PATH,
    SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME,
    SIMULATIONS_METRICS_BELADY_MIN_HIT_RATE_NAME,
    SIMULATIONS_METRICS_BELADY_MIN_MISS_RATE_NAME,
    SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME,
    SIMULATIONS_METRICS_HIT_RATE_NAME,
    SIMULATIONS_METRICS_MISS_RATE_NAME,
)
from src.const import (
    CACHE_LSTM_NAME,
    DATA_DYNAMIC_MODE,
    DATA_STATIC_MODE,
    DATASET_TESTING_SPLIT_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
    SIMULATIONS_METRICS_POLICY_NAME,
    SIMULATIONS_METRICS_TIMELINE_NAME,
)


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
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        dvc=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_SIMULATIONS,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        pipeline_config = prepare_pipeline_config()
        initialize_logs(
            logging.getLevelName(pipeline_config.logs.level),
            GrafanaLokiHandler(),
        )
        initialize_ray(
            pipeline_config.resources.general.num_cpus,
            pipeline_config.resources.general.num_gpus,
        )

        # Prepare configuration
        data_mode = pipeline_config.data.general.mode
        mistake_window = (
            pipeline_config.evaluation.simulations.metrics.mistake_rate.window
        )
        testing_batch_size = pipeline_config.data_loader.testing.batch_size
        testing_shuffle = pipeline_config.data_loader.testing.shuffle
        cache_size = pipeline_config.simulations.caches.dimension
        seed = pipeline_config.seed.value

        # Ensure reproducibility
        set_seed(seed)

        # Define cache eviction policies to simulate
        cache_eviction_policies = {
            CACHE_LRU_NAME: CacheWrapper(
                LRUCache,
                CacheMetricsLogger(),
                pipeline_config,
            ),
            CACHE_LFU_NAME: CacheWrapper(
                LFUCache,
                CacheMetricsLogger(),
                pipeline_config,
            ),
            CACHE_RANDOM_NAME: RandomCache(
                None,
                CacheMetricsLogger(),
                pipeline_config,
            ),
            CACHE_LSTM_NAME: LSTMCache(
                None,
                CacheMetricsLogger(),
                pipeline_config,
            ),
        }

        # Get testing set
        testing_set, _ = initialize_data_loader(
            DATASET_PROCESSED_TYPE,
            DATASET_TESTING_SPLIT_TYPE,
            testing_batch_size,
            testing_shuffle,
            AccessLogsDataset,
            pipeline_config,
        )

        info(
            "Simulations started",
            extra={
                "data_mode": data_mode,
                "policies_simulated": list(cache_eviction_policies.keys()),
                "mistake_window": mistake_window,
                "context": "Simulations",
            },
        )

        # For each cache eviction policy run a simulation
        # as a remote task via Ray
        results = []
        futures = []
        for policy, cache in cache_eviction_policies.items():
            futures.append(
                run_cache_simulation_task.remote(
                    cache,
                    policy,
                    testing_set,
                    pipeline_config,
                ),
            )

        # Retrieve all the simulation results
        for policy, future in zip(cache_eviction_policies.keys(), futures):
            counters, timeline, cache_latencies = ray.get(future)

            # Calculate salient metrics
            hit_rate, miss_rate, eviction_mistake_rate, avg_cache_latency = (
                calculate_simulation_metrics(
                    counters,
                    cache_latencies,
                    mistake_window,
                    cache_eviction_policies[policy].metrics_logger,
                )
            )

            # Collect all metrics together
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
            results.append(metrics)

            # Experiment tracking
            mlflow.log_metrics(
                {
                    "requests_num": counters[
                        SIMULATIONS_METRICS_HIT_COUNTER_NAME
                    ]
                    + counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                    "hits_num": counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME],
                    "misses_num": counters[
                        SIMULATIONS_METRICS_MISS_COUNTER_NAME
                    ],
                    "hit_rate": hit_rate,
                    "miss_rate": miss_rate,
                    "eviction_mistake_rate": eviction_mistake_rate,
                    "latency_us_min": min(cache_latencies),
                    "latency_us_max": max(cache_latencies),
                    "latency_us_avg": avg_cache_latency,
                    "latency_us_std": np.std(cache_latencies),
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

        # Determine plot file path according
        # to data distribution mode
        if data_mode == DATA_STATIC_MODE:
            plot_save_path = PLOT_STATIC_HIT_MISS_RATES_FILE_PATH
        elif data_mode == DATA_DYNAMIC_MODE:
            plot_save_path = PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH
        else:
            plot_save_path = PLOT_REAL_HIT_MISS_RATES_FILE_PATH

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
        mlflow.log_params(
            prepare_pipeline_config().model_dump(),
        )
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
            "plot_save_path": str(plot_save_path),
            "context": "Simulations",
        },
    )


if __name__ == "__main__":
    run_simulations()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
