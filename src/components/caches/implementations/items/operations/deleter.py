"""deleter.py

Module for removing items from cache data structures.

This module provides the `delete_item_from_cache` function, which deletes
a given key from the cache data and optionally from an associated frequency
data structure.

Functions:
    delete_item_from_cache(
        data: Any,
        key: int,
        freq_data: Any | None = None
    ) -> None
        Deletes the key from cache and frequency data if provided.
"""

from typing import Any

from mypy.checkexpr import defaultdict

from components.logs.levels.error_logger import error


def delete_item_from_cache(
    data: Any,
    key: int,
    freq_data: defaultdict | None = None,
) -> None:
    """Delete a key and its associated item from the cache data.

    This function deletes the key from the provided cache data. If a
    frequency data structure is provided, it also removes the key from there.

    Args:
        data (Any): Cache data.
        key (int): Key to delete.
        freq_data (defaultdict | None): Optional frequency data structure to
                                        delete the key from.

    Returns:
        None

    Raises:
        RuntimeError: If deletion of the key from cache or frequency
                      data fails:
            * The key does not exist in the cache data (KeyError).
            * The cache data structure is invalid or uninitialized
              (AttributeError).
    """
    try:
        # Delete the key from main cache
        del data[key]

        # Delete from frequency data if provided
        if freq_data is not None:
            freq_data.pop(key, None)
    except (AttributeError, KeyError) as e:
        msg = "Item removal from cache failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "key": key,
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
                "context": "Item removal from cache",
            },
        )
        raise RuntimeError(msg) from e
