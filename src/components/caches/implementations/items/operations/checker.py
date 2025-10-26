from typing import Any

from components.logs.levels.error_logger import error


def check_item_into_cache(data: Any, key: Any) -> bool:
    """Check if a key exists in a cache.

    This function, given a key, returns True if it exists in the
    provided cache data, False otherwise.

    Args:
        data (Any): Cache data.
        key (Any): Key to check for existence.

    Returns:
        bool: True if key exists in the cache, False otherwise.

    Raises:
        RuntimeError: If checking key existence fails:
            * The cache data is uninitialized or invalid (AttributeError).
            * The key is not hashable and cannot be used in the cache (TypeError).
    """
    try:
        return key in data
    except (AttributeError, TypeError) as e:
        msg = "Item existence in cache check failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "data_type": type(data).__name__ if data is not None else None,
                "data_size": (
                    len(data) if hasattr(data, "__len__") and data else 0
                ),
                "key_type": type(key).__name__,
                "context": "Item in cache existence check",
            },
        )
        raise RuntimeError(msg) from e
