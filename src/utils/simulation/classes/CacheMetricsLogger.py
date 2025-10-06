from collections import defaultdict

from utils.logs.levels.info_logger import info


class CacheMetricsLogger:
    """
    Logger class for tracing cache metrics.

    This class tracks:
        - Keys inserted into the cache (put events)
        - Keys accessed from the cache (get events)
        - Keys evicted from the cache
        - Keys predicted for prefetching
    """

    def __init__(self: "CacheMetricsLogger") -> None:
        """
        Initialize the CacheMetricsLogger class.

        This function sets up the data structures to store
        cache events and logs initialization start and success.

        Parameters:
            self ("CacheMetricsLogger"): Current class instance.

        Returns:
            None
        """
        # Initialize data structures to
        # keep track of cache events
        self.put_events = {}
        self.access_events = defaultdict(list)
        self.evicted_keys = defaultdict(list)
        self.prefetch_predictions = {}

        info("CacheMetricsLogger initialized")

    def log_put(
        self: "CacheMetricsLogger", key: int, time: float, ttl: float
    ) -> None:
        """
        Trace a key insertion into the cache.

        This function keep tracks of a key insertion
        into the cache, by storing the key and its
        expiration time.

        Parameters:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key inserted into the cache.
            time (float): Timestamp of insertion.
            ttl (float): Time-to-Live for the key.

        Returns:
            None
        """
        # Track put event
        self.put_events.setdefault(key, []).append((time, ttl))

        info(f"Key insertion traced for key: {key}")

    def log_get(self: "CacheMetricsLogger", key: int, time: float) -> None:
        """
        Trace key access from the cache.

        This function keep tracks of key access
        from the cache, by storing the key and the
        current time.

        Parameters:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key accessed from the cache.
            time (float): Timestamp of access.

        Returns:
            None
        """
        # Track get event
        self.access_events[key].append(time)

        info(f"Key access traced for key: {key}")

    def log_eviction(
        self: "CacheMetricsLogger", key: int, time: float
    ) -> None:
        """
        Trace a key eviction from the cache.

        This function keep tracks of key eviction
        from the cache, by storing the key and the
        current time.

        Parameters:
            self ("CacheMetricsLogger"): Current class instance.
            key (int): Key evicted from the cache.
            time (float): Timestamp of eviction.

        Returns:
            None
        """
        # Track eviction event
        self.evicted_keys[key].append(time)

        info(f"Key eviction traced for key: {key}")
