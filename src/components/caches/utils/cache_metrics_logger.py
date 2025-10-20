from collections import defaultdict
from typing import Any

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


class CacheMetricsLogger:
    """
    Logger class for tracing cache metrics.

    This class tracks cache events, including:
        - Keys inserted into the cache (put events)
        - Keys accessed from the cache (get events)
        - Keys evicted from the cache

    Attributes:
        put_events (Dict[int, List[Tuple[float, float]]]): Records key insertions
                                                           with timestamp and TTL.
        access_events (defaultdict(list)): Records access timestamps for each key.
        evicted_keys (defaultdict(list)): Records eviction timestamps for each key.
    """

    def __init__(self: "CacheMetricsLogger") -> None:
        """
        Initialize the CacheMetricsLogger class.

        This function sets up the data structures to store
        cache events and logs initialization start and success.

        Args:
            self ("CacheMetricsLogger"): Current class instance.

        Returns:
            None
        """
        # Initialize data structures to
        # keep track of cache events
        self.put_events = {}
        self.access_events = defaultdict(list)
        self.evicted_keys = defaultdict(list)

        info("CacheMetricsLogger initialized")

    def log_put(
        self: "CacheMetricsLogger", key: Any, time: float, ttl: float
    ) -> None:
        """
        Trace a key insertion into the cache.

        This function keep tracks of a key insertion into the cache,
        by storing the key and its expiration time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (Any): Key inserted into the cache.
            time (float): Timestamp of insertion.
            ttl (float): Time-to-Live for the key.

        Returns:
            None

        Raises:
            RuntimeError: If tracing the put event fails:
                * The internal put events dictionary is not initialized (AttributeError).
                * The provided key is not hashable and cannot be used as a dictionary
                  key (TypeError).
        """
        try:
            # Trace put event
            self.put_events.setdefault(key, []).append((time, ttl))
            debug(f"Key insertion traced for key: {key}")
        except (AttributeError, TypeError) as e:
            msg = "Failed to trace key insertion"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def log_get(self: "CacheMetricsLogger", key: Any, time: float) -> None:
        """
        Trace key access from the cache.

        This function keep tracks of key access from the cache, by storing
        the key and the current time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (Any): Key accessed from the cache.
            time (float): Timestamp of access.

        Returns:
            None

        Raises:
            RuntimeError: If tracing the get event fails:
                * The internal access events dictionary is not initialized
                  (AttributeError).
                * The provided key is not hashable and cannot be used as a
                  dictionary key (TypeError).
                * The key does not exist in access events dictionary (KeyError).
        """
        try:
            # Trace get event
            self.access_events[key].append(time)
            debug(f"Key access traced for key: {key}")
        except (AttributeError, TypeError, KeyError) as e:
            msg = "Failed to trace key access"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def log_eviction(
        self: "CacheMetricsLogger", key: Any, time: float
    ) -> None:
        """
        Trace a key eviction from the cache.

        This function keep tracks of key eviction
        from the cache, by storing the key and the
        current time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (Any): Key evicted from the cache.
            time (float): Timestamp of eviction.

        Returns:
            None

        Raises:
            RuntimeError: If tracing the eviction event fails:
                * The internal evicted keys dictionary is not initialized
                  (AttributeError).
                * The provided key is not hashable and cannot be used as a
                  dictionary key (TypeError).
                * The key does not exist in evicted keys (KeyError).
        """
        try:
            # Trace get event
            self.evicted_keys[key].append(time)
            debug(f"Key eviction traced for key: {key}")
        except (AttributeError, TypeError, KeyError) as e:
            msg = "Failed to trace key eviction"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
