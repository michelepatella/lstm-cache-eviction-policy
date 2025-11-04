"""popper.py

Module for removing items from cache data structures.

This module provides the `pop_item_from_cache` function, which removes
a key from the cache data, optionally updates the frequency data, and
returns the removed item.

Functions:
    pop_item_from_cache(
        data: Any,
        key: int,
        freq_data: Any | None = None
    ) -> Any | None
        Removes a cache item and optionally updates frequency tracking.
"""

from typing import Any

from mypy.checkexpr import defaultdict

from components.logs.levels.error_logger import error


def pop_item_from_cache(
    data: Any,
    key: int,
    freq_data: defaultdict | None = None,
) -> Any | None:
    """Remove an item from the cache data.

    This function removes the given key from the cache data and returns its
    associated item. Optionally, a callback can be executed after popping
    the item.

    Args:
        data (Any): Cache data.
        key (int): Key to remove.
        freq_data (defaultdict | None): Frequency data structure.

    Returns:
        Any | None: Item associated with the key removed
        (None if key not found).

    Raises:
        RuntimeError: If popping the item from cache fails:
            * The cache data structure is uninitialized or invalid
              (AttributeError).
            * The key is not hashable and cannot be used in the cache
              (TypeError).
    """
    try:
        # Remove key from the cache and get its item
        # (None if the key is not found)
        item = data.pop(key, None)

        # Remove from frequency data if applicable
        if freq_data is not None and item is not None:
            freq_data.pop(key, None)

        return item
    except (AttributeError, TypeError) as e:
        msg = "Item popping from cache failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "key_type": type(key).__name__,
                "data_type": type(data).__name__ if data is not None else None,
                "data_size": (
                    len(data) if hasattr(data, "__len__") and data else 0
                ),
                "freq_data_type": (
                    type(freq_data).__name__ if freq_data is not None else None
                ),
                "freq_data_len": (
                    len(freq_data)
                    if hasattr(freq_data, "__len__") and freq_data
                    else 0
                ),
                "freq_data_present": freq_data is not None,
                "context": "Item popping from cache",
            },
        )
        raise RuntimeError(msg) from e
