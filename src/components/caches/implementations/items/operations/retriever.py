from typing import Any

from components.logs.levels.error_logger import error


def get_item_from_cache(data: Any, key: Any) -> Any:
    """
    Retrieve an item from the cache data.

    This function retrieves the item associated with the provided key
    from the given cache data.

    Args:
        data (Any): Cache data.
        key (Any): Key to retrieve from the cache.

    Returns:
        Any: The item associated with the provided key.

    Raises:
        RuntimeError: If retrieving the cache item fails:
            * The key is not found in the cache (KeyError).
            * The cache data structure is invalid or uninitialized
              (AttributeError, TypeError).
    """
    try:
        # Get item by key
        return data[key]
    except (KeyError, AttributeError, TypeError) as e:
        msg = f"Failed to get item from cache"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
