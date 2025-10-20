import random
from typing import Any

from components.caches.implementations.items.operations.inserter import (
    insert_item_into_cache,
)
from components.caches.implementations.utils.base_cache import BaseCache
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


class RandomCache(BaseCache):
    """
    Random cache implementation.

    Evicts a random item when the maximum size is reached.
    """

    def _evict_random(self: "RandomCache", current_time: float) -> None:
        """
        Evict a random key from the cache.

        This function randomly selects a key stored in the cache,
        removes it along with its expiration, and logs the eviction.

        Args:
            self ("RandomCache"): Current class instance.
            current_time (float): Current timestamp for TTL management.

        Returns:
            None
        """
        try:
            # Select a random key to evict
            evict_key = random.choice(list(self.store.keys()))

            # Remove the selected key from cache
            self.store.pop(evict_key, None)
            self.expiry.pop(evict_key, None)

            debug(f"Random cache full, evicted key: {evict_key}")

            # Trace event
            self.metrics_logger.log_eviction(evict_key, current_time)
        except (IndexError, AttributeError, TypeError) as e:
            msg = "Failed to evict random key from cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def put(self: "RandomCache", key: Any, current_time: float) -> None:
        """
        Insert or update a key in the Random cache.

        This function inserts a key into the cache, updates its expiration
        time if already cached, evicts a random key if the cache is full,
        and logs the operation in the metrics logger.

        Args:
            self ("RandomCache"): Current class instance.
            key (Any): Key to insert or update.
            current_time (float): Current timestamp for TTL management.

        Returns:
            None

        Raises:
            RuntimeError: If eviction of a random key fails:
                * No keys are available to evict (IndexError).
                * The cache or expiry data structure is invalid or uninitialized
                  (AttributeError, TypeError).
        """
        try:
            debug(
                f"Key to put into random cache: {key}, "
                f"at time: {current_time}"
            )

            # Remove expired keys before insertion
            self._remove_expired_keys(current_time)

            # Insert item into cache with eviction
            # and TTL tracking
            insert_item_into_cache(
                data=self.store,
                key=key,
                item=key,
                cache_maxsize=self.maxsize,
                eviction_callback=lambda: self._evict_random(current_time),
                post_insert_callback=lambda k: self.expiry.__setitem__(
                    k, current_time + self.ttl
                ),
            )

            # Trace put event in metrics logger
            self.metrics_logger.log_put(key, current_time, self.ttl)

            debug(
                f"Random cache key inserted/updated: {key}, "
                f"expiration time: {self.expiry[key]}"
            )
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            msg = "Failed to put key into random cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
