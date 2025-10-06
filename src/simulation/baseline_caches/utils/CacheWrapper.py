from utils.simulation.BaseCache import BaseCache
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error


class CacheWrapper(BaseCache):
    """
    Generic cache wrapper.

    This class extends BaseCache to provide a standard
    cache insertion mechanism with TTL and metrics logging.
    """

    def put(self: "CacheWrapper", key: int, current_time: float) -> None:
        """
        Insert a key into the cache.

        This function inserts a key into the cache, removes
        expired entries, updates the expiration time, and logs
        the operation in the metrics logger.

        Parameters:
            self ("CacheWrapper"): Current class instance.
            key (int): Key to insert.
            current_time (float): Current timestamp.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs during cache insertion e.g.:
                * Attribute access failure.
                * Invalid types for key or current time.
                * Unexpected internal error.
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

            debug(
                f"Key inserted into cache: {key}, "
                f"expiration time: {self.expiry[key]}"
            )
        except (AttributeError, TypeError) as e:
            msg = "Failed to insert key into cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
