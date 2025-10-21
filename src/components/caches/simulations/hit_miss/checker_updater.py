from typing import Any, Dict

from components.logs.levels.error_logger import error
from components.logs.levels.debug_logger import debug
from const import HIT_COUNTER_NAME, MISS_COUNTER_NAME


def check_update_hit_miss(
    cache: Any,
    key: Any,
    current_time: float,
    counters: Dict[str, int],
    hit_counter_name: str = HIT_COUNTER_NAME,
    miss_counter_name: str = MISS_COUNTER_NAME,
) -> bool:
    """
    Check key presence in the cache and update hit and miss counters.

    This function checks whether a given key exists in the cache
    and is still valid based on its expiration time. It updates
    the hit/miss counters accordingly.

    Args:
        cache (Any): Cache object implementing a contains method.
        key (Any): Key to look up in the cache.
        current_time (float): Current timestamp for validation.
        counters (Dict[str, int]): Dictionary containing cache hit/miss counters.
        hit_counter_name (str): Name of cache hit counter.
        miss_counter_name (str): Name of cache miss counter.

    Returns:
        bool: True if the key is found in the cache, False otherwise.

    Raises:
        RuntimeError: If checking and updating hit and miss counters fails:
            * The cache object is invalid or lacks required methods
              (AttributeError, TypeError).
            * The counters dictionary is invalid or missing keys
              (AttributeError, KeyError).
    """
    try:
        # Check whether the cache contains the key
        if cache.contains(key, current_time):
            # Increase hit counter by one
            counters[hit_counter_name] += 1

            debug(f"Cache HIT — key: {key}, time: {current_time}")

            return True

        # Increase miss counter by one
        counters[miss_counter_name] += 1

        debug(f"Cache MISS — key: {key}, time: {current_time}")

        return False
    except (AttributeError, TypeError, KeyError) as e:
        msg = "Failed to check and update cache hit and miss counters"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
