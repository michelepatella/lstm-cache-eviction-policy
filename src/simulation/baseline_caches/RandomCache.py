import random

from utils.simulation.classes.BaseCache import BaseCache
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error


class RandomCache(BaseCache):
    """
    Random cache implementation.

    This cache evicts a random item when the
    maximum size is reached. Each cached key
    also has an expiration time based on the
    configured TTL (Time-To-Live).
    """

    def put(self: "RandomCache", key: int, current_time: float) -> None:
        """
        Insert or update a key in the Random cache.

        This function inserts a key into the cache, updates
        its expiration time if already cached, and evicts
        a random key if the cache is full.

        Parameters:
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

            # Check if key already exists in cache
            if self.contains(key, current_time):
                # Update TTL
                self.expiry[key] = current_time + self.ttl

                debug(
                    f"Random cache key already cached updated: {key}, "
                    f"new expiration time: {self.expiry[key]}"
                )

                # Trace event
                self.metrics_logger.log_put(key, current_time, self.ttl)

                # Exit
                return

            # Check whether the cache is full
            if len(self.store) >= self.maxsize:
                # Evict a random key among those
                # stored in the cache
                evict_key = random.choice(list(self.store.keys()))

                # Evict selected key
                self.store.pop(evict_key)
                self.expiry.pop(evict_key)

                debug(f"Random cache full, evicted key: {evict_key}")

                # Trace event
                self.metrics_logger.log_eviction(evict_key, current_time)

            # Insert new key along with
            # its expiration time
            self.store[key] = key
            self.expiry[key] = current_time + self.ttl

            debug(
                f"Random cache key inserted: {key}, "
                f"expiration time: {self.expiry[key]}"
            )

            # Trace event
            self.metrics_logger.log_put(key, current_time, self.ttl)
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            msg = "Random cache put failed"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
