from typing import Any, Callable, Optional

from components.logs.levels.error_logger import error


def insert_item_into_cache(
    data: Any,
    key: Any,
    item: Any,
    cache_maxsize: float,
    eviction_callback: Callable,
    pre_insert_callback: Optional[Callable] = None,
    post_insert_callback: Optional[Callable] = None,
) -> None:
    """
    Insert or update an item into a cache structure with eviction support.

    This function handles insertion logic for multiple cache strategies, performing
    eviction if the cache is full, applying optional pre- and post-insert callbacks.

    Args:
        data (Any): Cache data.
        key (Any): Key to insert or update.
        item (Any): Value associated with the key.
        cache_maxsize (float): Maximum cache size.
        eviction_callback (Callable): Callback to evict one item from the cache.
        pre_insert_callback (Optional[Callable]): Action to perform before the insertion.
        post_insert_callback (Optional[Callable]): Action to perform after the insertion.

    Returns:
        None

    Raises:
        RuntimeError: If inserting the item into cache fails:
            * The cache data structure is invalid or uninitialized
              (AttributeError, TypeError).
    """
    try:
        # If cache full and key not present then evict one
        if len(data) >= cache_maxsize and key not in data:
            eviction_callback()

        # Optional pre-insert action
        if pre_insert_callback:
            pre_insert_callback(key)

        # Actual insertion
        data[key] = item

        # Optional post-insert action
        if post_insert_callback:
            post_insert_callback(key)
    except (TypeError, AttributeError) as e:
        msg = "Failed to insert item into cache"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
