from abc import ABC, abstractmethod
from typing import Any

from config.classes.Config import Config
from utils.simulation.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


class BaseCache(ABC):
    """
    Abstract base class for all cache implementations.

    This class provides common functionalities such as:
    - TTL (Time-To-Live) management
    - Expired key removal
    - Metrics logging
    """

    def __init__(
        self: "BaseCache",
        cache_class: Any,
        metrics_logger: CacheMetricsLogger,
        config: Config,
    ) -> None:
        """
        Initialize the BaseCache.

        This function sets up the cache, metrics logger,
        TTL, maximum size, and auxiliary data structures.

        Parameters:
            self (BaseCache): Current class instance.
            cache_class (Any): Class implementing a cache.
            metrics_logger (CacheMetricsLogger): Object to log cache
                                                 events.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Prepare configuration
        dimension = config.simulation.general.dimension
        ttl = config.simulation.general.ttl

        debug(f"BaseCache dimension: {dimension}")
        debug(f"BaseCache TTL: {ttl}")

        # Initialize cache and fields
        self.cache = (
            cache_class(
                dimension,
                callback=self._on_evict,
            )
            if cache_class is not None
            else None
        )
        self.maxsize = dimension
        self.ttl = ttl
        self.metrics_logger = metrics_logger
        self.store = {}
        self.expiry = {}
        self.scores = {}
        self._last_put_time = None

        info("BaseCache initialized")

    def _is_expired(self: "BaseCache", key: int, current_time: float) -> bool:
        """
        Check if a key has expired based on its TTL.

        This function, given a key and the current time,
        checks whether a key has expired, based on the
        TTL. The function returns True if the key
        has expired, False otherwise.

        Parameters:
            self (BaseCache): Current class instance.
            key (int): Key to check expiration for.
            current_time (float): Current timestamp.

        Returns:
            bool: True if key has expired, False otherwise.
        """
        # Check whether the key has expired,
        # provided it is stored in the cache
        expired = key in self.expiry and self.expiry[key] < current_time

        debug(
            f"Key {key} expired BaseCache check: {expired},"
            f" at time: {current_time}"
        )

        return expired

    def _remove_expired_keys(self: "BaseCache", current_time: float) -> None:
        """
        Remove all expired keys from the cache.

        This function, given the current time, removes
        all the expired keys from the cache, based on TTL.

        Parameters:
            self (BaseCache): Current class instance.
            current_time (float): Current timestamp.

        Returns:
            None
        """
        # Identify expired keys
        expired_keys = [
            k for k, exp in self.expiry.items() if exp < current_time
        ]

        debug(f"Expired keys to be removed from BaseCache: {expired_keys}")

        # Remove all the expired keys
        for k in expired_keys:
            # Remove key from cache/store
            if self.cache is not None:
                self.cache.pop(k, None)
            else:
                self.store.pop(k, None)

            # Remove TTL and scores (if any)
            self.expiry.pop(k, None)
            if self.scores is not None:
                self.scores.pop(k, None)

            # Trace eviction
            self.metrics_logger.log_eviction(k, current_time)

            debug(
                f"Expired key removed from BaseCache: {k},"
                f" at time: {current_time}"
            )

    def contains(self: "BaseCache", key: int, current_time: float) -> bool:
        """
        Check if a key exists in the cache and is
        not expired.

        This function, given a key and the current time,
        checks whether a key exists in the cache and is not
        expired, returning True if both condition are verified,
        False otherwise.

        Parameters:
            self (BaseCache): Current class instance.
            key (int): Key to check.
            current_time (float): Current timestamp.

        Returns:
            bool: True if key exists and is valid,
                  False otherwise.
        """
        debug(f"Key containment in BaseCache check for: {key}")

        # Trace the get event
        self.metrics_logger.log_get(key, current_time)

        # Check whether the key is
        # in the cache/store and is not expired
        in_cache = (
            self.cache is not None
            and key in self.cache
            and not self._is_expired(key, current_time)
        ) or (key in self.store and not self._is_expired(key, current_time))

        debug(f"Key {key} in BaseCache (and valid): {in_cache}")

        return in_cache

    def _on_evict(self: "BaseCache", key: int) -> None:
        """
        Callback triggered when a key is evicted
        from BaseCache.

        This function represents a callback triggered
        when a key is evicted from BaseCache. The
        callback removes expiration time for the evicted
        key as well as logs the eviction event.

        Parameters:
            self (BaseCache): Current class instance.
            key (int): Key that was evicted.

        Returns:
            None
        """
        # Remove expiration time
        # of the evicted key
        self.expiry.pop(key, None)

        # Trace eviction event
        self.metrics_logger.log_eviction(key, self._last_put_time)

        debug(f"Key {key} evicted from BaseCache")

    @abstractmethod
    def put(self: "BaseCache", *args: Any, **kwargs: Any) -> None:
        """
        Insert a key into the cache.

        This function represents an abstract method
        managing key items insertion in the cache.
        Each cache eviction strategy must implement
        its own put method.

        Parameters:
            self ("BaseCache"): Current class instance.
            *args (Any): Positional arguments required
                         by the specific cache strategy.
            **kwargs (Any): Keyword arguments required
                            by the specific cache strategy.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not
                                 implemented by the subclass.
        """
        raise NotImplementedError()
