"""runner.py

Module for running cache simulations with different eviction policies.

This module provides the `run_cache_simulation` function, which
simulates cache behavior over a testing dataset, tracks hit/miss
counters, maintains a timeline of hits/misses, and measures
cache access latencies for analysis.

Functions:
    run_cache_simulation(
        cache: Any,
        policy: str,
        testing_set: AccessLogsDataset,
        pipeline_config: PipelineConfig
    ) -> tuple[dict[str, int], list[dict[str, float]], list[float]]
        Executes the cache simulation for the given cache and policy.
"""

import time
from typing import Any

from tqdm import tqdm

from components.caches.simulations.hit_miss.checker_updater import (
    check_update_hit_miss,
)
from components.caches.simulations.hit_miss.timeline_updater import (
    update_hit_miss_timeline,
)
from components.const import LIST_FIRST_IDX, TIME_MICROSECONDS_IN_SECOND
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.time.transforms.trig_decoder import (
    decode_time_trigonometrically,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from src.const import (
    CACHE_LSTM_NAME,
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
)


def run_cache_simulation(
    cache: Any,
    policy: str,
    testing_set: AccessLogsDataset,
    pipeline_config: PipelineConfig,
) -> tuple[dict[str, int], list[dict[str, float]], list[float]]:
    """Run a full cache simulation for a given cache eviction policy.

    This function runs a cache simulation over the testing dataset, managing
    requests one by one according to the provided eviction policy (either
    LSTM-based or baseline). It collects key metrics such as hit/miss counters,
    timeline evolution, and cache access latencies.

    Args:
        cache (Any): Cache object implementing the eviction policy.
        policy (str): Cache eviction policy name.
        testing_set (AccessLogsDataset): Access logs dataset.
        pipeline_config (PipelineConfig): Configuration object.

    Returns:
        tuple[dict[str, int], list[dict[str, float]], list[float]]:
            - counters: Dictionary containing hit and miss counts.
            - timeline: List of dictionaries showing the evolution of hits
                        and misses over time.
            - cache_latencies: List of cache access latencies in microseconds.

    Raises:
    RuntimeError: If simulating the cache policy fails:
        * Accessing the testing set by index fails due to an invalid
          index or data structure (IndexError, TypeError).
        * Unpacking the dataset row fails due to unexpected row format or
          type (ValueError, TypeError).
        * Extracting features or target from the dataset row fails due to
          missing or malformed elements (ValueError, TypeError,
          AttributeError).
    """
    try:
        info(
            "Cache simulation started",
            extra={
                "policy": policy,
                "cache_type": type(cache).__name__ if cache else None,
                "context": f"{policy} cache simulation",
            },
        )

        # Initialize data
        counters = {
            SIMULATIONS_METRICS_HIT_COUNTER_NAME: 0,
            SIMULATIONS_METRICS_MISS_COUNTER_NAME: 0,
        }
        timeline = []
        cache_latencies = []

        # Iterate over testing set, assuming each
        # row represents a request to be satisfied
        tqdm_bar = tqdm(range(len(testing_set)), desc=f"{policy}")
        for idx in tqdm_bar:
            # Extract the current row from the dataset
            row = testing_set[idx]

            # Extrapolate current time and requested
            # key from the current row
            features, _, target = row

            # Decode time
            sin_time, cos_time, _, _ = features[LIST_FIRST_IDX]
            current_time = decode_time_trigonometrically(sin_time, cos_time)

            # Decode key
            key = target.item()

            # Start timer to keep track of cache latency
            start_time = time.perf_counter()

            # Check whether the requested key
            # is into the baseline cache
            is_hit = check_update_hit_miss(cache, key, current_time, counters)

            # If the requested key is not into the cache
            if not is_hit:
                if policy == CACHE_LSTM_NAME:
                    # Put the requested key into the LSTM cache
                    cache.put(
                        key,
                        current_time,
                        idx,
                        testing_set,
                        pipeline_config,
                    )
                else:
                    # Put the requested key into the baseline cache
                    cache.put(key, current_time)

            # Stop timer to keep track of cache latency
            end_time = time.perf_counter()

            # Store cache latency
            cache_latency = (
                end_time - start_time
            ) * TIME_MICROSECONDS_IN_SECOND
            cache_latencies.append(cache_latency)

            # Update number of hits and misses
            timeline = update_hit_miss_timeline(idx, counters, timeline)

            # To update bar real-time
            hit_rate = (
                counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME]
                / max(
                    1,
                    counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME]
                    + counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                )
            ) * 100

            miss_rate = (
                counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME]
                / max(
                    1,
                    counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME]
                    + counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                )
            ) * 100

            tqdm_bar.set_postfix(
                hit_rate=f"{hit_rate:.2f}%",
                miss_rate=f"{miss_rate:.2f}%",
            )

        info(
            "Cache simulation completed",
            extra={
                "policy": policy,
                "cache_type": type(cache).__name__ if cache else None,
                "requests_num": len(testing_set),
                "hits_num": counters[SIMULATIONS_METRICS_HIT_COUNTER_NAME],
                "misses_num": counters[SIMULATIONS_METRICS_MISS_COUNTER_NAME],
                "context": f"{policy} cache simulation",
            },
        )

        return counters, timeline, cache_latencies
    except (TypeError, IndexError, ValueError, AttributeError) as e:
        msg = "Cache simulation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "policy": policy,
                "cache_type": type(cache).__name__ if cache else None,
                "context": f"{policy} cache simulation",
            },
        )
        raise RuntimeError(msg) from e
