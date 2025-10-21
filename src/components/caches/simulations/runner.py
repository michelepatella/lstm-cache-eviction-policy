import time
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from components.caches.simulations.hit_miss.checker_updater import (
    check_update_hit_miss,
)
from components.caches.simulations.hit_miss.timeline_updater import (
    update_hit_miss_timeline,
)
from components.const import TIME_MICROSECONDS_IN_SECOND
from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.time.transforms.trig_decoder import (
    decode_time_trigonometrically,
)
from src.const import (
    CACHE_LSTM_NAME,
    DATASET_TESTING_SPLIT_TYPE,
    SIMULATIONS_METRICS_HIT_COUNTER_NAME,
    SIMULATIONS_METRICS_MISS_COUNTER_NAME,
)
from pipeline.config.pydantic.config import Config


def run_cache_simulation(
    cache: Any,
    policy: str,
    config: Config,
) -> Tuple[Dict[str, int], List[Dict[str, float]], List[float]]:
    """
    Run a full cache simulation for a given cache eviction policy.

    This function runs a cache simulation over the testing dataset, managing
    requests one by one according to the provided eviction policy (either
    LSTM-based or baseline). It collects key metrics such as hit/miss counters,
    timeline evolution, and cache access latencies.

    Args:
        cache (Any): Cache object implementing the eviction policy.
        policy (str): Cache eviction policy name.
        config (Config): Configuration object.

    Returns:
        Tuple[Dict[str, int], List[Dict[str, float]], List[float]]:
            - counters: Dictionary containing hit and miss counts.
            - timeline: List of dictionaries showing the evolution of hits and misses
                        over time.
            - cache_latencies: List of cache access latencies in microseconds.

    Raises:
    RuntimeError: If simulating the cache policy fails:
        * Accessing the testing set by index fails due to an invalid
          index or data structure (IndexError, TypeError).
        * Unpacking the dataset row fails due to unexpected row format or
          type (ValueError, TypeError).
        * Extracting features or target from the dataset row fails due to
          missing or malformed elements (ValueError, TypeError, AttributeError).
    """
    try:
        info(f"{policy} simulation started")

        # Prepare configuration
        testing_batch_size = config.testing.general.batch_size
        testing_shuffle = config.testing.general.shuffle

        # Initialize data
        counters = {
            SIMULATIONS_METRICS_HIT_COUNTER_NAME: 0,
            SIMULATIONS_METRICS_MISS_COUNTER_NAME: 0,
        }
        timeline = []
        cache_latencies = []

        # Get testing set
        testing_set, testing_loader = initialize_data_loader(
            DATASET_TESTING_SPLIT_TYPE,
            testing_batch_size,
            testing_shuffle,
            AccessLogsDataset,
            config,
        )

        # Iterate over testing set, assuming each
        # row represents a request to be satisfied
        for idx in tqdm(
            range(len(testing_set)),
            desc=f"{policy}",
        ):
            # Extract the current row from the dataset
            row = testing_set[idx]

            # Extrapolate current time and requested
            # key from the current row
            features, _, target = row

            # Decode time
            sin_time, cos_time = features[0]
            current_time = decode_time_trigonometrically(sin_time, cos_time)

            # Decode key
            key = target.item()

            debug(f"Row {idx} (key={key}, current_time={current_time})")

            # Start timer to keep track of cache latency
            start_time = time.perf_counter()

            # Check whether the requested key
            # is into the baseline cache
            is_hit = check_update_hit_miss(cache, key, current_time, counters)

            # If the requested key is not into the cache
            if not is_hit:
                if policy == CACHE_LSTM_NAME:
                    # Put the requested key into the LSTM cache
                    cache.put(key, current_time, idx, testing_set, config)
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

        info(f"{policy} simulation completed")

        return counters, timeline, cache_latencies
    except (TypeError, IndexError, ValueError, AttributeError) as e:
        msg = "Failed to simulate cache policy"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
