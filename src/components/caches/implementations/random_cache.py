import random

from components.caches.utils.base_cache import BaseCache
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
        # Select a random key to evict
        evict_key = random.choice(list(self.store.keys()))

        # Remove the selected key from cache
        self.store.pop(evict_key, None)
        self.expiry.pop(evict_key, None)

        debug(f"Random cache full, evicted key: {evict_key}")

        # Trace event
        self.metrics_logger.log_eviction(evict_key, current_time)

    def put(self: "RandomCache", key: int, current_time: float) -> None:
        """
        Insert or update a key in the Random cache.

        This function inserts a key into the cache, updates
        its expiration time if already cached, and evicts
        a random key if the cache is full.

        Args:
            self ("RandomCache"): Current class instance.
            key (int): Key to insert or update.
            current_time (float): Current timestamp for
                                  TTL management.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs while putting
                          a key into the random cache e.g.:
                            * If some fields don't exist.
                            * If key or current time are
                              of wrong type.
                            * If a non-existent key is popped.
                            * If trying to select a random key
                              from an empty list.
        """
        try:
            # Remove expired keys before insertion
            self._remove_expired_keys(current_time)

            # Check whether the cache is full and
            # the key is new
            if len(self.store) >= self.maxsize and key not in self.store:
                # Evict a random key
                self._evict_random(current_time)

            # (Re)Insert new key along with
            # its expiration time
            self.store[key] = key
            self.expiry[key] = current_time + self.ttl

            debug(
                f"Random cache key inserted/updated: {key}, "
                f"expiration time: {self.expiry[key]}"
            )

            # Trace event
            self.metrics_logger.log_put(key, current_time, self.ttl)
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            msg = "Random cache put failed"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
