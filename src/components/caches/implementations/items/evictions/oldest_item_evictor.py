from collections.abc import Callable
from typing import Any

from components.logs.levels.error_logger import error


def evict_oldest_item(
    data: Any,
    callback: Callable | None,
) -> tuple[Any, Any]:
    """Evict the oldest item from a cache.

    This function removes the oldest key and its associated item from
    the given cache data. If a callback is provided, it is invoked with
    the evicted key.

    Args:
        data (Any): Cache data.
        callback (Callable | None): Callback function invoked with the
                                    evicted key.

    Returns:
        tuple[Any, Any]:
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
                "oldest_key_type": (
                    type(oldest_key).__name__
                    if "oldest_key" in locals()
                    else None
                ),
                "oldest_item_type": (
                    type(oldest_item).__name__
                    if "oldest_item" in locals()
                    else None
                ),
                "callback_present": callback is not None,
                "context": "Oldest item eviction",
            },
        )
        raise RuntimeError(msg) from e
