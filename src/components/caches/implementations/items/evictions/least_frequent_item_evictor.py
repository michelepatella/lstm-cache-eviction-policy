from collections import defaultdict
from typing import Any, Callable, Optional, Tuple

from components.logs.levels.error_logger import error


def evict_least_frequent_item(
    data: Any, data_freq: defaultdict, callback: Optional[Callable]
) -> Tuple[Any, int]:
    """
    Evict the least frequently used item from a cache.

    This function identifies and removes the least frequently used key
    from the given cache data and its associated frequency dictionary.
    The tiebreak strategy consists of evicting the oldest key among
    those with the same minimum frequency. If a callback is provided,
    it is invoked with the evicted key.

    Args:
        data (Any): Cache data structure.
        data_freq (defaultdict): Dictionary tracking access frequencies
                                 of cached items.
        callback (Optional[Callable]): Callback function invoked with the
                                       evicted key.

    Returns:
        Tuple[Any, int]:
            - key_to_evict: The evicted key.
            - min_freq: Frequency value of evicted key before eviction.

    Raises:
        RuntimeError: If eviction of the least frequent item fails:
            * The frequency dictionary is empty when attempting to find
              the minimum frequency (ValueError).
            * No candidates exist with the minimum frequency (IndexError).
            * The key to evict is not found in the cache or frequency
              dictionary (KeyError).
            * The cache or frequency data structures are uninitialized or
              invalid (AttributeError).
            * The data structures do not support comparison or deletion
              operations (TypeError).
    """
    try:
        # Identify least frequency
        min_freq = min(data_freq.values())

        # Identify candidate keys with minimum frequency
        evicted_candidates = [k for k, f in data_freq.items() if f == min_freq]

        # Tiebreak strategy: select the oldest
        # key among them
        key_to_evict = evicted_candidates[0]

        # Remove selected key from cache
        # and frequency dictionary
        del data[key_to_evict]
        del data_freq[key_to_evict]

        # Callback if present
        if callback:
            callback(key_to_evict)

        return key_to_evict, min_freq
    except (
        ValueError,
        IndexError,
        KeyError,
        AttributeError,
        TypeError,
    ) as e:
        msg = "Failed to evict the least frequent item"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
