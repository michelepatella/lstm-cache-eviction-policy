"""cache_wrapper.py

Module for a generic cache wrapper extending BaseCache.

This module provides the `CacheWrapper` class, a standard cache
implementation that handles key insertions with TTL management,
expired key removal, and metrics logging. It serves as a base
for simple cache use cases or testing of eviction strategies.

Classes:
    CacheWrapper
        Generic cache wrapper implementing put method with TTL
        and metrics logging.
"""

from components.caches.implementations.utils.base_cache import BaseCache
from components.logs.levels.error_logger import error


class CacheWrapper(BaseCache):
    """Generic cache wrapper that extends BaseCache.

    This class provides a standard cache insertion mechanism
    with TTL (Time-To-Live) management and metrics logging.
    """

    def put(self: "CacheWrapper", key: int, current_time: float) -> None:
        """Insert a key into the cache.

        This function inserts a key into the cache, removes expired entries,
        updates the expiration time, and logs the operation in the metrics
        logger.

        Args:
            self ("CacheWrapper"): Current class instance.
            key (int): Key to insert.
            current_time (float): Current timestamp.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion into cache wrapper fails:
                * The cache or supporting structures are not initialized
                 (AttributeError).
                * The provided key is not hashable (TypeError).
        """
        try:
            # Track last insertion timestamp
            self._last_put_time = current_time

            # Remove expired keys before
            # inserting new one
            self._remove_expired_keys(current_time)

            # Insert key into the cache
            self.cache[key] = key

            # Update expiration time for the key
            self.expiry[key] = current_time + self.ttl

            # Log cache operation in metrics logger
            self.metrics_logger.log_put(key, current_time, self.ttl)
        except (AttributeError, TypeError) as e:
            msg = "Item insertion into cache wrapper failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "current_time": current_time,
                    "ttl": getattr(self, "ttl", None),
                    "cache_keys_count": (
                        len(self.cache)
                        if hasattr(self, "cache") and self.cache
                        else None
                    ),
                    "expiry_keys_count": (
                        len(self.expiry)
                        if hasattr(self, "expiry") and self.expiry
                        else None
                    ),
                    "metrics_logger_initialized": hasattr(
                        self,
                        "metrics_logger",
                    ),
                    "cache_wrapper_class": type(self).__name__,
                    "context": "CacheWrapper",
                },
            )
            raise RuntimeError(msg) from e
