"""cache_size_calculator.py

Utility function for calculating cache size.

This module provides the `calculate_cache_size` function which returns
the current number of items stored in a cache data structure.

Functions:
    calculate_cache_size(data: Any) -> int
        Returns the number of items currently stored in the cache.
"""

from typing import Any

from components.logs.levels.error_logger import error


def calculate_cache_size(data: Any) -> int:
    """Get the number of items currently in a cache.

    This function returns the number of items currently stored in
    the provided cache data structure.

    Args:
        data (Any): Cache data.

    Returns:
        int: Number of items in the cache.

    Raises:
        RuntimeError: If accessing the cache size fails:
            * The cache data structure is uninitialized or invalid
             (AttributeError).
            * The cache data structure does not support the method
              to calculate its size (TypeError).
    """
    try:
        # Calculate cache size
        cache_size = len(data)

        return cache_size
    except (AttributeError, TypeError) as e:
        msg = "Cache size calculatation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "data_type": type(data).__name__ if data is not None else None,
                "context": "Cache size calculation",
            },
        )
        raise RuntimeError(msg) from e
