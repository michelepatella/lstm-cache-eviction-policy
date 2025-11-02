import random
from http.client import HTTPException
from typing import Any

import pandas as pd
import requests
from box import Box

from components.caches.implementations.utils.base_cache import BaseCache
from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.const import (
    API_ENDPOINT,
    API_PARAM_KEYS_IN_CACHE_NAME,
    API_PARAM_LAST_ACCESSES_NAME,
    API_PARAM_USER_API_KWARGS_NAME,
)
from components.dataset.rows.extractions.lasts_extractor import (
    extract_last_rows_from_dataset,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
        config: Any,
    ) -> None:
        """Initialize the LSTM cache.

        This function initializes the LSTM cache by
        calling the BaseCache constructor.

        Args:
            self ("LSTMCache"): Current class instance.
            cache_class (Any): Underlying cache class
                               to store items.
            metrics_logger (CacheMetricsLogger): Logger for cache events.
            config (Any): Configuration object.

        Returns:
            None
        """
        # Cache class initialization
        super().__init__(cache_class, metrics_logger, config)

        # Set API kwargs to use
        self._api_kwargs = config.simulations.api_kwargs

        debug(
            "Cache initialization executed",
            extra={
                "maxsize": self.maxsize,
                "api_kwargs": self._api_kwargs.__dict__,
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
        testing_set: pd.DataFrame,
        config: Any,
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
            testing_set (pd.DataFrame): Testing dataset for sequence extraction.
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion/eviction fails:
                * Key is unhashable or cache store/expiry dict
                  misconfigured (TypeError, AttributeError).
                * API call fails (HTTPException).
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
                    testing_set.data,
                )

                # Check whether last accesses
                # are not available
                if last_accesses is None:
                    # Eviction fallback policy: Random
                    key_to_evict = random.choice(list(self.store.keys()))
                else:
                    # Call API to get
                    # the key to be evicted from the cache
                    response = requests.post(
                        API_ENDPOINT,
                        json={
                            API_PARAM_KEYS_IN_CACHE_NAME: list(
                                self.store.keys(),
                            ),
                            API_PARAM_LAST_ACCESSES_NAME: last_accesses,
                            API_PARAM_USER_API_KWARGS_NAME: self._api_kwargs.__dict__,
                        },
                    )
                    data = Box(response.json())

                    # Extract the key(s) from API response
                    # as well as the kwargs used
                    key_to_evict = list(data.keys_to_evict)

                # Evict key(s)
                for key in key_to_evict:
                    self.evict_key(key)

                    # Track eviction event
                    self.metrics_logger.log_eviction(key, current_time)

            # Insert the key
            self._put_key(key, current_time)

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
