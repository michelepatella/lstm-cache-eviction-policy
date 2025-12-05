"""simulator.py

Module defining a Ray remote task for parallelized cache simulation runs.

This module provides the `run_cache_simulation_task` function, which wraps the
cache simulation logic to be executed asynchronously via the Ray framework.
This enables efficient, parallel evaluation of different cache policies and
configurations against a test dataset.

Functions:
    run_cache_simulation_task(
        cache: Any,
        policy: str,
        testing_set: AccessLogsDataset,
        pipeline_config: PipelineConfig
    ) -> tuple[dict[str, int], list[dict[str, float]], list[float]]
        Remote task to execute a full cache simulation for a specific policy.
"""

from typing import Any

import ray

from components.caches.simulations.runner import run_cache_simulation
from components.dataset.access_logs_dataset import AccessLogsDataset
from pipeline.config.pydantic.pipeline_config import PipelineConfig


@ray.remote
def run_cache_simulation_task(
    cache: Any,
    policy: str,
    testing_set: AccessLogsDataset,
    pipeline_config: PipelineConfig,
) -> tuple[dict[str, int], list[dict[str, float]], list[float]]:
    """Remote task to run a full cache simulation using a specified policy.

    This function is executed remotely via Ray. It calls the `run_cache_simulation`
    utility, performing a sequence of accesses on the provided cache instance
    and recording performance metrics.

    Args:
        cache (Any): The cache object to be simulated.
        policy (str): The name of the eviction policy being tested.
        testing_set (AccessLogsDataset): The access log dataset used to drive
                                         the simulation.
        pipeline_config (PipelineConfig): The configuration object.

    Returns:
        tuple[dict[str, int], list[dict[str, float]], list[float]]:
            - counters: Dictionary containing hit and miss counts.
            - timeline: List of dictionaries showing the evolution of hits
                        and misses over time.
            - cache_latencies: List of cache access latencies in microseconds.
    """
    return run_cache_simulation(cache, policy, testing_set, pipeline_config)
