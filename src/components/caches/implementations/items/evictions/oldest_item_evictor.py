"""oldest_item_evictor.py

Module for evicting the oldest item from a cache.

This module provides the `evict_oldest_item` function, which removes the least
recently used item from a cache data structure. An optional callback can be
executed with the evicted key.

Functions:
    evict_oldest_item(
        data: Any,
        callback: (...) -> Any | None
    ) -> tuple[int, Any]
        Evicts the oldest item from the cache.
"""

from collections.abc import Callable
from typing import Any

from components.logs.levels.error_logger import error


def evict_oldest_item(
    data: Any,
    callback: Callable | None,
) -> tuple[int, Any]:
    """Evict the oldest item from a cache.

    This function removes the oldest key and its associated item from
    the given cache data. If a callback is provided, it is invoked with
    the evicted key.

    Args:
        data (Any): Cache data.
        callback (Callable | None): Callback function invoked with the
                                    evicted key.

    Returns:
        tuple[int, Any]:
            - oldest_key: The key of the evicted item.
            - oldest_item: The value of the evicted item.

    Raises:
        RuntimeError: If eviction of the oldest item fails:
            * The cache is empty when attempting to evict an item (KeyError).
            * The cache data structure is invalid or uninitialized
              (AttributeError, TypeError).
    """
    try:
        # Remove the oldest key and item
        # from cache data
        oldest_key, oldest_item = data.popitem(last=False)

        # Callback if present
        if callback:
            callback(oldest_key)

        return oldest_key, oldest_item
    except (KeyError, AttributeError, TypeError) as e:
        msg = "Oldest item eviction failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "data_type": type(data).__name__ if data is not None else None,
                "data_size": (
                    len(data) if hasattr(data, "__len__") and data else 0
                ),
                "callback_present": callback is not None,
                "context": "Oldest item eviction",
            },
        )
        raise RuntimeError(msg) from e
