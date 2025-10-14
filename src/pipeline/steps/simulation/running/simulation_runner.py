import time
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from pipeline.config.classes.Config import Config
from pipeline.const import LSTM_CACHE_NAME, MICROSECONDS_IN_SECOND
from pipeline.const import (
    HIT_COUNTER_NAME,
    MISS_COUNTER_NAME,
    TESTING_SPLIT_TYPE,
)
from utils.data_loader.building.initializer import initialize_data_loader
from pipeline.steps.simulation.running.utils.hit_miss_checker_updater import (
    check_update_hit_miss,
)
from pipeline.steps.simulation.running.utils.hit_miss_timeline_updater import (
    update_hit_miss_timeline,
)
from pipeline.steps.simulation.utils.time_key_from_row_extractor import (
    extract_time_key_from_row,
)
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.error_logger import error


def run_cache_simulation(
    cache: Any,
    policy: str,
    config: Config,
) -> Tuple[Dict[str, int], List[Dict[str, float]], List[float]]:
    """
    Run a full cache simulation for a given cache
    eviction policy.

    This function runs a cache simulation over the
    testing dataset, managing requests one by one
    according to the provided eviction policy (either
    LSTM-based or baseline). It collects key metrics
    such as hit/miss counters, timeline evolution,
    and cache access latencies.

    Args:
        cache (Any): Cache object implementing the eviction policy.
        policy (str): Cache eviction policy name.
        config (Config): Configuration object.

    Returns:
        Tuple[Dict[str, int], List[Dict[str, float]], List[float]]:
            A tuple containing counters for hits and misses, a timeline
            of hits and misses evolution, and a list of caches latencies.

    Raises:
        RuntimeError: If an error occurs during the cache simulation, e.g.:
            * The cache object does not implement required methods.
            * The input dataset row structure is invalid.
            * Required dictionary keys are missing.
    """
    # Prepare configuration
    testing_batch_size = config.testing.general.batch_size
    testing_shuffle = config.testing.general.shuffle

    # Initialize data
    counters = {
        HIT_COUNTER_NAME: 0,
        MISS_COUNTER_NAME: 0,
    }
    timeline = []
    cache_latencies = []

    # Get testing set
    testing_set, testing_loader = initialize_data_loader(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        AccessLogsDataset,
        config,
    )

    try:
        # Iterate over testing set, assuming each
        # row represents a request to be satisfied
        for idx in tqdm(
            range(len(testing_set)),
            desc=f"Simulating {policy}",
        ):
            # Extract the current row
            # from the dataset
            row = testing_set[idx]

            # Extrapolate current time and requested
            # key from the current row
            current_time, key = extract_time_key_from_row(row)

            # Start timer to keep track of
            # cache latency
            start_time = time.perf_counter()

            # Check whether the requested key
            # is into the baseline cache
            is_hit = check_update_hit_miss(cache, key, current_time, counters)

            # If the requested key is not
            # into the cache
            if not is_hit:
                if policy == LSTM_CACHE_NAME:
                    # Put the requested key
                    # into the LSTM cache
                    cache.put(key, current_time, idx, testing_set, config)
                else:
                    # Put the requested key
                    # into the baseline cache
                    cache.put(key, current_time)

            # Stop timer to keep track of
            # cache latency
            end_time = time.perf_counter()

            # Store cache latency
            cache_latency = (end_time - start_time) * MICROSECONDS_IN_SECOND
            cache_latencies.append(cache_latency)

            # Update number of hits and misses
            timeline = update_hit_miss_timeline(idx, counters, timeline)
    except (TypeError, KeyError, AttributeError) as e:
        msg = "Cache simulation failed"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    return counters, timeline, cache_latencies
