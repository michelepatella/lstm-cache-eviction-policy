from typing import Any, Dict

from pipeline.const import HIT_COUNTER_NAME, MISS_COUNTER_NAME
from components.logs.levels.info_logger import info


def check_update_hit_miss(
    cache: Any, key: int, current_time: float, counters: Dict[str, int]
) -> bool:
    """
    Check key presence in the cache
    and update hit and miss counters.

    This function checks whether a given key exists in the cache
    and is still valid based on its expiration time. It updates
    the hit/miss counters accordingly.

    Args:
        cache (Any): Cache object implementing a contains method.
        key (int): Key to look up in the cache.
        current_time (float): Current timestamp for validation.
        counters (Dict[str, int]): Dictionary containing cache
                                   hit/miss counters.

    Returns:
        bool: True if the key is found in the cache, False otherwise.
    """
    # Check whether the cache contains the key
    if cache.contains(key, current_time):
        # Increase hit counter by one
        counters[HIT_COUNTER_NAME] += 1

        info(f"Cache HIT — Key: {key}, Time: {current_time}")

        return True

    # Increase miss counter by one
    counters[MISS_COUNTER_NAME] += 1

    info(f"Cache MISS — Key: {key}, Time: {current_time}")

    return False
