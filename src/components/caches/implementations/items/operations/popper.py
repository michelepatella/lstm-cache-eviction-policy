from typing import Any, Optional

from mypy.checkexpr import defaultdict

from components.logs.levels.error_logger import error


def pop_item_from_cache(
    data: Any,
    key: Any,
    freq_data: Optional[defaultdict] = None,
) -> Optional[Any]:
    """
    Remove an item from the cache data.

    This function removes the given key from the cache data and returns its
    associated item. Optionally, a callback can be executed after popping the item.

    Args:
        data (Any): Cache data.
        key (Any): Key to remove.
        freq_data (Optional[defaultdict]): Frequency data structure.

    Returns:
        Optional[Any]: Item associated with the key removed (None if key not found).

    Raises:
        RuntimeError: If popping the item from cache fails:
            * The cache data structure is uninitialized or invalid (AttributeError).
            * The key is not hashable and cannot be used in the cache (TypeError).
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
        msg = "Failed to pop item from cache"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
