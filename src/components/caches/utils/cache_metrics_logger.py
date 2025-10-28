from collections import defaultdict

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


class CacheMetricsLogger:
    """Logger class for tracing cache metrics.

    This class tracks cache events, including:
        - Keys inserted into the cache (put events)
        - Keys accessed from the cache (get events)
        - Keys evicted from the cache

    Attributes:
        put_events (Dict[int, List[Tuple[float, float]]]):
            Records key insertions with timestamp and TTL.
        access_events (defaultdict(list)): Records access timestamps
                                           for each key.
        evicted_keys (defaultdict(list)): Records eviction timestamps
                                          for each key.
    """

    def __init__(self: "CacheMetricsLogger") -> None:
        """Initialize the CacheMetricsLogger class.

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

        debug(
            "CacheMetricsLogger initialization executed",
            extra={"context": "CacheMetricsLogger"},
        )

    def log_put(
        self: "CacheMetricsLogger",
        key: int,
        time: float,
        ttl: float,
    ) -> None:
        """Trace a key insertion into the cache.

        This function keep tracks of a key insertion into the cache,
        by storing the key and its expiration time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key inserted into the cache.
            time (float): Timestamp of insertion.
            ttl (float): Time-to-Live for the key.

        Returns:
            None

        Raises:
            RuntimeError: If tracing the put event fails:
                * The internal put events dictionary is not initialized
                  (AttributeError).
                * The provided key is not hashable and cannot be used as
                  a dictionary key (TypeError).
        """
        try:
            # Trace put event
            self.put_events.setdefault(key, []).append((time, ttl))
        except (AttributeError, TypeError) as e:
            msg = "Tracing item insertion failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "time": time,
                    "ttl": ttl,
                    "put_events_initialized": hasattr(self, "put_events"),
                    "current_put_events_count": (
                        len(self.put_events)
                        if hasattr(self, "put_events")
                        else None
                    ),
                    "cache_class": type(self).__name__,
                    "context": "CacheMetricsLogger",
                },
            )
            raise RuntimeError(msg) from e

    def log_get(self: "CacheMetricsLogger", key: int, time: float) -> None:
        """Trace key access from the cache.

        This function keep tracks of key access from the cache, by storing
        the key and the current time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key accessed from the cache.
            time (float): Timestamp of access.

        Returns:
            None

        Raises:
            RuntimeError: If tracing the get event fails:
                * The internal access events dictionary is not initialized
                  (AttributeError).
                * The provided key is not hashable and cannot be used as a
                  dictionary key (TypeError).
                * The key does not exist in access events dictionary
                  (KeyError).
        """
        try:
            # Trace get event
            self.access_events[key].append(time)
        except (AttributeError, TypeError, KeyError) as e:
            msg = "Tracing item access failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "time": time,
                    "access_events_initialized": hasattr(
                        self,
                        "access_events",
                    ),
                    "current_access_events_count": (
                        len(self.access_events)
                        if hasattr(self, "access_events")
                        else None
                    ),
                    "cache_class": type(self).__name__,
                    "context": "CacheMetricsLogger",
                },
            )
            raise RuntimeError(msg) from e

    def log_eviction(
        self: "CacheMetricsLogger",
        key: int,
        time: float,
    ) -> None:
        """Trace a key eviction from the cache.

        This function keep tracks of key eviction
        from the cache, by storing the key and the
        current time.

        Args:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key evicted from the cache.
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
        except (AttributeError, TypeError, KeyError) as e:
            msg = "Tracing item eviction failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "time": time,
                    "evicted_keys_initialized": hasattr(self, "evicted_keys"),
                    "current_evicted_keys_count": (
                        len(self.evicted_keys)
                        if hasattr(self, "evicted_keys")
                        else None
                    ),
                    "cache_class": type(self).__name__,
                    "context": "CacheMetricsLogger",
                },
            )
            raise RuntimeError(msg) from e
