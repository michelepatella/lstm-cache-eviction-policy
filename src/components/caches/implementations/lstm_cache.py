import random
from http.client import HTTPException
from typing import Any

import requests
from box import Box
from torch.utils.data import DataLoader

from components.caches.implementations.utils.base_cache import BaseCache
from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.const import (
    EVICTION_POLICY_API_ENDPOINT,
    EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME,
    EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME,
    EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME,
)
from components.dataset.rows.extractions.lasts_extractor import (
    extract_last_rows_from_dataset,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config


class LSTMCache(BaseCache):
    """LSTM-based cache implementation.

    Evicts keys from the cache based on an LSTM
    eviction policy when the cache is full.

    Attributes:
        _api_kwargs (dict): Keyword arguments used by the eviction
                            policy API.
    """

    def __init__(
        self: "LSTMCache",
        cache_class: Any,
        metrics_logger: CacheMetricsLogger,
        config: Config,
    ) -> None:
        """Initialize the LSTM cache.

        This function initializes the LSTM cache by
        calling the BaseCache constructor.

        Args:
            self ("LSTMCache"): Current class instance.
            cache_class (Any): Underlying cache class
                               to store items.
            metrics_logger (CacheMetricsLogger): Logger for cache events.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Cache class initialization
        super().__init__(cache_class, metrics_logger, config)

        self._api_kwargs = {}

        debug(
            "Cache initialization executed",
            extra={
                "maxsize": self.maxsize,
                "context": "LSTM cache",
            },
        )

    def evict_key(self: "LSTMCache", key: int) -> None:
        """Evict a key from the cache.

        This function evicts a provided key
        from the LSTM cache, along with its
        expiration time.

        Args:
            self ("LSTMCache"): Current class instance.
            key (int): Key to remove from the cache.

        Returns:
            None

        Raises:
            RuntimeError: If key eviction from LSTM cache fails:
                * The key is unhashable (TypeError).
                * Cache store or expiry dict is misconfigured (AttributeError).
        """
        try:
            # Remove key from store and its
            # expiration time
            self.store.pop(key, None)
            self.expiry.pop(key, None)
        except (TypeError, AttributeError) as e:
            msg = "Key eviction from LSTM cache failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "store_keys": list(self.store.keys())
                    if hasattr(self.store, "keys")
                    else None,
                    "expiry_keys": list(self.expiry.keys())
                    if hasattr(self.expiry, "keys")
                    else None,
                    "context": "LSTM cache",
                },
            )
            raise RuntimeError(msg) from e

    def _put_key(self: "LSTMCache", key: int, current_time: float) -> None:
        """Put a key in the cache.

        This function puts a key into the LSTM cache
        along with its expiration time.

        Args:
            self ("LSTMCache"): Current class instance.
            key (int): Key to insert in the cache.
            current_time (float): Current time.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion into LSTM cache fails:
                * The key is unhashable (TypeError).
                * Cache store or expiry dict is misconfigured (AttributeError).
        """
        try:
            # Insert key in the store along
            # with its expiration time
            self.store[key] = key
            self.expiry[key] = current_time + self.ttl

            # Track put event
            self.metrics_logger.log_put(key, current_time, self.ttl)
        except (TypeError, AttributeError) as e:
            msg = "Key insertion into LSTM cache failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "current_time": current_time,
                    "store_keys": list(self.store.keys())
                    if hasattr(self.store, "keys")
                    else None,
                    "expiry_keys": list(self.expiry.keys())
                    if hasattr(self.expiry, "keys")
                    else None,
                    "context": "LSTM cache",
                },
            )
            raise RuntimeError(msg) from e

    def put(
        self: "LSTMCache",
        key: int,
        current_time: float,
        current_idx: int,
        testing_set: DataLoader,
        config: Config,
    ) -> None:
        """Insert a key in the LSTM cache.

        This function puts a key into the LSTM cache
        along with its expiration time. If the cache is full,
        extracts the last sequence of accesses and uses the
        LSTM policy to decide which key to evict.

        Args:
            self ("LSTMCache"): Current class instance.
            key (int): Key to insert.
            current_time (float): Current time.
            current_idx (int): Index of the current request.
            testing_set (DataLoader): Testing dataset for sequence extraction.
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion/eviction fails:
                * Key is unhashable or cache store/expiry dict
                  misconfigured (TypeError, AttributeError).
                * Eviction policy API call fails (HTTPException).
        """
        try:
            # Remove all expired keys
            # before insertion
            self._remove_expired_keys(current_time)

            api_kwargs = None
            # Check whether the cache is full
            if key not in self.store and len(self.store) >= self.maxsize:
                # Get the sequence length
                # of the LSTM model
                seq_len = config.model.sequence.length

                # Extract last accesses of
                # sequence length
                last_accesses = extract_last_rows_from_dataset(
                    current_idx,
                    seq_len,
                    testing_set,
                )

                # Check whether last accesses
                # are not available
                if last_accesses is None:
                    # Eviction fallback policy: Random
                    key_to_evict = random.choice(list(self.store.keys()))
                else:
                    # Call eviction policy API to get
                    # the key to be evicted from the cache
                    response = requests.post(
                        EVICTION_POLICY_API_ENDPOINT,
                        json={
                            EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME: list(
                                self.store.keys(),
                            ),
                            EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME: last_accesses,
                            EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME: {},
                        },
                    )
                    data = Box(response.json())

                    # Extract the key(s) from API response
                    # as well as the kwargs used
                    key_to_evict = list(data.keys_to_evict)
                    api_kwargs = dict(data.kwargs)

                # Evict key(s)
                for key in key_to_evict:
                    self.evict_key(key)

                    # Track eviction event
                    self.metrics_logger.log_eviction(key, current_time)

            # Insert the key
            self._put_key(key, current_time)

            # Keep track of API kwargs used
            if api_kwargs is not None:
                self._api_kwargs = api_kwargs

        except (TypeError, AttributeError, HTTPException) as e:
            msg = "Key insertion/eviction in/from LSTM cache failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "current_time": current_time,
                    "store_keys": list(self.store.keys())
                    if hasattr(self.store, "keys")
                    else None,
                    "expiry_keys": list(self.expiry.keys())
                    if hasattr(self.expiry, "keys")
                    else None,
                    "context": "LSTM cache",
                },
            )
            raise RuntimeError(msg) from e
