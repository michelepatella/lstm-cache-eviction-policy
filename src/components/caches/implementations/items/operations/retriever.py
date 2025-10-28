from typing import Any

from components.logs.levels.error_logger import error


def get_item_from_cache(data: Any, key: int) -> Any:
    """Retrieve an item from the cache data.

    This function retrieves the item associated with the provided key
    from the given cache data.

    Args:
        data (Any): Cache data.
        key (int): Key to retrieve from the cache.

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
        msg = "Item retrieval from cache failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "key_type": type(key).__name__,
                "data_type": type(data).__name__ if data is not None else None,
                "data_size": (
                    len(data) if hasattr(data, "__len__") and data else 0
                ),
                "context": "Item retrieval from cache",
            },
        )
        raise RuntimeError(msg) from e
