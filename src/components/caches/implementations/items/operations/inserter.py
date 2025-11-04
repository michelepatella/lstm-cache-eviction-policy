"""inserter.py

Module for inserting items into cache data structures with eviction support.

This module provides the `insert_item_into_cache` function, which inserts
or updates a key-value pair in a cache, performs eviction if the cache is
full, and optionally executes pre- and post-insert callbacks.

Functions:
    insert_item_into_cache(
        data: Any,
        key: int,
        item: Any,
        cache_maxsize: float,
        eviction_callback: (...) -> Any,
        pre_insert_callback: (...) -> Any | None = None,
        post_insert_callback: (...) -> Any | None = None
    ) -> None
        Inserts or updates a cache item and handles eviction logic.
"""

from collections.abc import Callable
from typing import Any

from components.logs.levels.error_logger import error


def insert_item_into_cache(
    data: Any,
    key: int,
    item: Any,
    cache_maxsize: float,
    eviction_callback: Callable,
    pre_insert_callback: Callable | None = None,
    post_insert_callback: Callable | None = None,
) -> None:
    """Insert or update an item into a cache structure with eviction support.

    This function handles insertion logic for multiple cache strategies,
    performing eviction if the cache is full, applying optional pre-
    and post-insert callbacks.

    Args:
        data (Any): Cache data.
        key (int): Key to insert or update.
        item (Any): Value associated with the key.
        cache_maxsize (float): Maximum cache size.
        eviction_callback (Callable): Callback to evict one item from
                                      the cache.
        pre_insert_callback (Callable | None): Action to perform before
                                                  the insertion.
        post_insert_callback (Callable | None): Action to perform after
                                                the insertion.

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
        msg = "Item insertion into cache failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "key_type": type(key).__name__,
                "item_type": type(item).__name__,
                "data_type": type(data).__name__ if data is not None else None,
                "data_size": (
                    len(data) if hasattr(data, "__len__") and data else 0
                ),
                "cache_maxsize": cache_maxsize,
                "eviction_callback_present": eviction_callback is not None,
                "pre_insert_callback_present": pre_insert_callback is not None,
                "post_insert_callback_present": post_insert_callback
                is not None,
                "context": "Item insertion into cache",
            },
        )
        raise RuntimeError(msg) from e
