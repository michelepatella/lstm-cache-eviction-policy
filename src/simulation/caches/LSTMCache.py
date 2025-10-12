import random
from typing import Any

import requests
from torch.utils.data import DataLoader

from config.classes.Config import Config
from const import EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME, EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME, \
    EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME, EVICTION_POLICY_API_ENDPOINT
from simulation.caches.utils.classes.BaseCache import BaseCache
from simulation.caches.utils.classes.CacheMetricsLogger import (
    CacheMetricsLogger,
)
from simulation.caches.utils.last_accesses_extractor import get_last_accesses
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


class LSTMCache(BaseCache):
    """
    LSTM-based cache implementation.

    Evicts keys from the cache based on an LSTM
    eviction policy when the cache is full.
    """

    def __init__(
        self: "LSTMCache",
        cache_class: Any,
        metrics_logger: CacheMetricsLogger,
        config: Config,
    ) -> None:
        """
        Initialize the LSTM cache.

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

        info("LSTM cache initialized")

    def evict_key(self: "LSTMCache", key: int) -> None:
        """
        Evict a key from the cache.

        This function evicts a provided key
        from the LSTM cache, along with its
        expiration time.

        Args:
            self ("LSTMCache"): Current class instance.
            key (int): Key to remove from the cache.

        Returns:
            None
        """
        # Remove key from store and its
        # expiration time
        self.store.pop(key, None)
        self.expiry.pop(key, None)

        debug(f"LSTM cache evicted key: {key}")

    def _put_key(self: "LSTMCache", key: int, current_time: float) -> None:
        """
        Put a key in the cache.

        This function puts a key into the LSTM cache
        along with its expiration time.

        Args:
            self ("LSTMCache"): Current class instance.
            key (int): Key to insert in the cache.
            current_time (float): Current time.

        Returns:
            None
        """
        # Insert key in the store along
        # with its expiration time
        self.store[key] = key
        self.expiry[key] = current_time + self.ttl

        # Track put event
        self.metrics_logger.log_put(key, current_time, self.ttl)

        debug(f"LSTM cache inserted key: {key}, at time {current_time}")

    def put(
        self: "LSTMCache",
        key: int,
        current_time: float,
        current_idx: int,
        testing_set: DataLoader,
        config: Config,
    ) -> None:
        """
        Insert a key in the LSTM cache.

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
        """
        # Remove all expired keys
        # before insertion
        self._remove_expired_keys(current_time)

        # Check whether the cache is full
        if key not in self.store and len(self.store) >= self.maxsize:
            # Get the sequence length
            # of the LSTM model
            seq_len = config.model.sequence.length

            # Extract last accesses of
            # sequence length
            last_accesses = get_last_accesses(
                current_idx, seq_len, testing_set
            )

            # Check whether last accesses
            # are not available
            if last_accesses is None:
                # Eviction fallback policy: Random
                key_to_evict = random.choice(list(self.store.keys()))
            else:
                # Call LSTM eviction policy API to get
                # the key to be evicted from the cache
                response = requests.post(
                    EVICTION_POLICY_API_ENDPOINT,
                    json={
                        EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME: list(self.store.keys()),
                        EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME: last_accesses,
                        EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME: {}
                    },
                )
                response.raise_for_status()

                # Extract the key from API response
                key_to_evict = ...

            # Evict key
            self.evict_key(key_to_evict)

            # Track eviction event
            self.metrics_logger.log_eviction(key_to_evict, current_time)

        # Insert the key
        self._put_key(key, current_time)
