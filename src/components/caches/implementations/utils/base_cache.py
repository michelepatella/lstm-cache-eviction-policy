from abc import ABC, abstractmethod
from typing import Any

from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config


class BaseCache(ABC):
    """Abstract base class for all cache implementations.

    Provides common functionalities such as:
    - TTL (Time-To-Live) management
    - Expired key removal
    - Metrics logging for get, put, and eviction events

    Attributes:
        cache (Any): Underlying cache instance.
        maxsize (int): Maximum number of keys allowed in the cache.
        ttl (float): Time-to-Live for each key.
        metrics_logger (CacheMetricsLogger): Logger for cache events.
        store (dict): Dictionary storing cache keys and values.
        expiry (dict): Dictionary storing expiration time per key.
        scores (dict): Dictionary storing scores for keys.
        _last_put_time (Optional[float]): Timestamp of the last put operation.
    """

    def __init__(
        self: "BaseCache",
        cache_class: Any,
        metrics_logger: CacheMetricsLogger,
        config: Config,
    ) -> None:
        """Initialize the BaseCache.

        This function sets up the cache, metrics logger, TTL, maximum size,
        and auxiliary data structures.

        Args:
            self (BaseCache): Current class instance.
            cache_class (Any): Class implementing a cache.
            metrics_logger (CacheMetricsLogger): Object to log cache events.
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If base cache initialization fails:
                * Invalid or incompatible cache class (TypeError).
                * Missing or invalid attributes in callback (AttributeError).
                * Invalid cache dimension or TTL (ValueError).
        """
        try:
            # Prepare configuration
            cache_dimension = config.simulations.cache.dimension
            ttl = config.simulations.cache.ttl

            # Initialize cache and fields
            self.cache = (
                cache_class(
                    cache_dimension,
                    callback=self._on_evict,
                )
                if cache_class is not None
                else None
            )
            self.maxsize = cache_dimension
            self.ttl = ttl
            self.metrics_logger = metrics_logger
            self.store = {}
            self.expiry = {}
            self.scores = {}
            self._last_put_time = None

            debug(
                "BaseCache initialization executed",
                extra={"context": "BaseCache"},
            )
        except (TypeError, AttributeError, ValueError) as e:
            msg = "Cache initialization failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "expiry_initialized": isinstance(
                        getattr(self, "expiry", None),
                        dict,
                    ),
                    "ttl": getattr(config.simulations.cache, "ttl", None),
                    "cache_class": (
                        str(cache_class) if cache_class is not None else None
                    ),
                    "cache_dimension": getattr(
                        config.simulations.cache,
                        "dimension",
                        None,
                    ),
                    "store_initialized": isinstance(
                        getattr(self, "store", None),
                        dict,
                    ),
                    "scores_initialized": isinstance(
                        getattr(self, "scores", None),
                        dict,
                    ),
                    "metrics_logger_type": type(metrics_logger).__name__,
                    "context": "BaseCache",
                },
            )
            raise RuntimeError(msg) from e

    def _is_expired(self: "BaseCache", key: Any, current_time: float) -> bool:
        """Check if a key has expired based on its TTL.

        This function, given a key and the current time, checks whether a
        key has expired, based on the TTL. The function returns True if the key
        has expired, False otherwise.

        Args:
            self (BaseCache): Current class instance.
            key (Any): Key to check expiration for.
            current_time (float): Current timestamp.

        Returns:
            bool: True if key has expired, False otherwise.

        Raises:
            RuntimeError: If checking expiration fails:
                * The key is not hashable (TypeError).
                * The key is not present in the expiry dictionary
                  (KeyError).
                * The values involved are not comparable or invalid
                  (ValueError).
        """
        try:
            # Check whether the key has expired,
            # provided it is stored in the cache
            return key in self.expiry and self.expiry[key] < current_time
        except (TypeError, KeyError, ValueError) as e:
            msg = "Cache item expiration check failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "key_type": type(key).__name__,
                    "current_time": current_time,
                    "expiry_type": (
                        type(self.expiry).__name__
                        if hasattr(self, "expiry")
                        else None
                    ),
                    "expiry_len": (
                        len(self.expiry)
                        if hasattr(self, "expiry") and self.expiry
                        else 0
                    ),
                    "context": "BaseCache",
                },
            )
            raise RuntimeError(msg) from e

    def _remove_expired_keys(self: "BaseCache", current_time: float) -> None:
        """Remove all expired keys from the cache.

        This function, given the current time, removes all the expired keys
        from the cache, based on TTL.

        Args:
            self (BaseCache): Current class instance.
            current_time (float): Current timestamp.

        Returns:
            None

        Raises:
            RuntimeError: If removing expired keys fails:
                * Cache object not supporting pop operation
                  (TypeError, AttributeError).
                * Metrics logger failing to log an eviction
                  (AttributeError, TypeError).
        """
        try:
            # Identify expired keys
            expired_keys = [
                k for k in self.expiry if self._is_expired(k, current_time)
            ]

            # Remove all the expired keys
            for k in expired_keys:
                # Remove key from cache/store
                if self.cache is not None:
                    self.cache.pop(k)
                else:
                    self.store.pop(k, None)

                # Remove TTL
                self.expiry.pop(k, None)

                # Remove score (if any)
                if self.scores is not None:
                    self.scores.pop(k, None)

                # Trace eviction
                self.metrics_logger.log_eviction(k, current_time)
        except (TypeError, AttributeError) as e:
            msg = "Cache expired key removal failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "current_time": current_time,
                    "expired_keys_num": (
                        len(expired_keys) if "expired_keys" in locals() else 0
                    ),
                    "expiry_type": (
                        type(self.expiry).__name__
                        if hasattr(self, "expiry")
                        else None
                    ),
                    "expiry_len": (
                        len(self.expiry)
                        if hasattr(self, "expiry") and self.expiry
                        else 0
                    ),
                    "scores_type": (
                        type(self.scores).__name__
                        if hasattr(self, "scores")
                        else None
                    ),
                    "scores_len": (
                        len(self.scores)
                        if hasattr(self.scores) and self.scores
                        else 0
                    ),
                    "store_type": (
                        type(self.store).__name__
                        if hasattr(self, "store")
                        else None
                    ),
                    "store_size": (
                        len(self.store)
                        if hasattr(self, "store") and self.store
                        else 0
                    ),
                    "context": "BaseCache",
                },
            )
            raise RuntimeError(msg) from e

    def contains(self: "BaseCache", key: Any, current_time: float) -> bool:
        """Check if a key exists in the cache and is not expired.

        This function, given a key and the current time, checks
        whether a key exists in the cache and is not expired, returning
        True if both condition are verified, False otherwise.

        Args:
            self (BaseCache): Current class instance.
            key (Any): Key to check.
            current_time (float): Current timestamp.

        Returns:
            bool: True if key exists and is valid, False otherwise.

        Raises:
            RuntimeError: If the key existence check fails:
                * The cache or store not being initialized (AttributeError).
                * Membership check failing (TypeError).
                * The metrics logger failing to log the get event
                  (AttributeError, TypeError).
        """
        try:
            # Trace the get event
            self.metrics_logger.log_get(key, current_time)

            # Check whether the key is
            # in the cache/store and is not expired
            in_cache = (
                self.cache is not None
                and key in self.cache
                and not self._is_expired(key, current_time)
            ) or (
                key in self.store and not self._is_expired(key, current_time)
            )

            return in_cache
        except (TypeError, AttributeError) as e:
            msg = "Cache item existence check failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "key_type": type(key).__name__,
                    "current_time": current_time,
                    "store_type": (
                        type(self.store).__name__
                        if hasattr(self, "store")
                        else None
                    ),
                    "store_size": (
                        len(self.store)
                        if hasattr(self, "store") and self.store
                        else 0
                    ),
                    "cache_type": (
                        type(self.cache).__name__
                        if hasattr(self, "cache") and self.cache
                        else None
                    ),
                    "cache_size": (
                        len(self.cache)
                        if hasattr(self, "cache") and self.cache
                        else 0
                    ),
                    "context": "BaseCache",
                },
            )
            raise RuntimeError(msg) from e

    def _on_evict(self: "BaseCache", key: Any) -> None:
        """Callback triggered when a key is evicted from BaseCache.

        This function represents a callback triggered when a key is evicted
        from BaseCache. The callback removes expiration time for the evicted
        key as well as logs the eviction event.

        Args:
            self (BaseCache): Current class instance.
            key (Any): Key that was evicted.

        Returns:
            None

        Raises:
            RuntimeError: If the eviction handling fails:
                * The expiry dictionary not being initialized (AttributeError).
                * The metrics logger not being initialized or failing to log
                  (AttributeError, TypeError).
                * The last put time being invalid or causing issues in logging
                  (TypeError, ValueError).
        """
        try:
            # Remove expiration time
            # of the evicted key
            self.expiry.pop(key, None)

            # Trace eviction event
            self.metrics_logger.log_eviction(key, self._last_put_time)
        except (TypeError, AttributeError, ValueError) as e:
            msg = "Cache item eviction failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "key_type": type(key).__name__,
                    "expiry_type": (
                        type(self.expiry).__name__
                        if hasattr(self, "expiry")
                        else None
                    ),
                    "expiry_len": (
                        len(self.expiry)
                        if hasattr(self, "expiry") and self.expiry
                        else 0
                    ),
                    "last_put_time": self._last_put_time,
                    "context": "BaseCache",
                },
            )
            raise RuntimeError(msg) from e

    @abstractmethod
    def put(self: "BaseCache", *args: Any, **kwargs: Any) -> None:
        """Insert a key into the cache.

        This function represents an abstract method managing key items
        insertion in the cache. Each cache eviction strategy must implement
        its own put method.

        Args:
            self ("BaseCache"): Current class instance.
            *args (Any): Positional arguments required by the specific
                         cache strategy.
            **kwargs (Any): Keyword arguments required by the specific
                            cache strategy.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented
                                 by the subclass.
        """
        raise NotImplementedError
