"""lr_cache.py

Module implementing a Logistic Regression-based cache.

This module provides the `LRCache` class, which manages key-value pairs in a
cache using a Logistic Regression-based eviction policy when the cache is full.

Classes:
    LRCache(cache_class, metrics_logger, config):
        Logistic Regression cache implementation supporting put, eviction, and
        key operations.
"""

import random
from typing import Any

import joblib
import numpy as np
import pandas as pd

from components.caches.implementations.utils.base_cache import BaseCache
from components.caches.utils.cache_metrics_logger import (
    CacheMetricsLogger,
)
from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    LIST_FIRST_IDX,
    MODEL_LR_TRAINED_DYNAMIC_FILE_PATH,
    MODEL_LR_TRAINED_STATIC_FILE_PATH,
    TENSOR_FEATURES_DIM,
)
from components.dataset.builder import build_dataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import (
    DATA_STATIC_MODE,
    DATASET_COLUMN_LR_PREVIOUS_REQUEST_PREFIX_NAME,
    DATASET_COLUMN_REQUEST_NAME,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig


class LRCache(BaseCache):
    """Logistic Regression-based cache implementation.

    Evicts keys from the cache based on a Logistic Regression-based
    eviction policy when the cache is full.
    """

    def __init__(
        self: "LRCache",
        cache_class: Any,
        metrics_logger: CacheMetricsLogger,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Initialize the Logistic Regression cache.

        This function initializes the Logistic Regression cache by
        calling the BaseCache constructor.

        Args:
            self ("LRCache"): Current class instance.
            cache_class (Any): Underlying cache class to store items.
            metrics_logger (CacheMetricsLogger): Logger for cache events.
            pipeline_config (PipelineConfig): Configuration object.

        Returns:
            None
        """
        # Cache class initialization
        super().__init__(cache_class, metrics_logger, pipeline_config)

        # Determine the path to load model from
        data_mode = pipeline_config.data.general.mode
        if data_mode == DATA_STATIC_MODE:
            model_save_path = MODEL_LR_TRAINED_STATIC_FILE_PATH
        else:
            model_save_path = MODEL_LR_TRAINED_DYNAMIC_FILE_PATH

        # Load model
        self.model = joblib.load(model_save_path)

        debug(
            "Cache initialization executed",
            extra={
                "maxsize": self.maxsize,
                "context": "LR Cache",
            },
        )

    def evict_key(self: "LRCache", key: int) -> None:
        """Evict a key from the cache.

        This function evicts a provided key from the Logistic Regression
        cache, along with its expiration time.

        Args:
            self ("LRCache"): Current class instance.
            key (int): Key to remove from the cache.

        Returns:
            None

        Raises:
            RuntimeError: If key eviction from Logistic Regression cache fails:
                * The key is unhashable (TypeError).
                * Cache store or expiry dict is misconfigured (AttributeError).
        """
        try:
            # Remove key from store and its
            # expiration time
            self.store.pop(key, None)
            self.expiry.pop(key, None)
        except (TypeError, AttributeError) as e:
            msg = "Key eviction from Logistic Regression cache failed"
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
                    "context": "LR Cache",
                },
            )
            raise RuntimeError(msg) from e

    def _put_key(
        self: "LRCache",
        key: int,
        current_time: float,
    ) -> None:
        """Put a key in the cache.

        This function puts a key into the Logistic Regression cache
        along with its expiration time.

        Args:
            self ("LRCache"): Current class instance.
            key (int): Key to insert in the cache.
            current_time (float): Current time.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion into Logistic Regression cache fails:
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
            msg = "Key insertion into Logistic Regression cache failed"
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
                    "context": "LR Cache",
                },
            )
            raise RuntimeError(msg) from e

    def put(
        self: "LRCache",
        key: int,
        current_time: float,
        current_idx: int,
        testing_set: pd.DataFrame,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Insert a key in the Logistic Regression cache.

        This function puts a key into the Logistic Regression cache
        along with its expiration time. If the cache is full, a
        Logistic Regression model is exploited for getting an eviction decision.

        Args:
            self ("LRCache"): Current class instance.
            key (int): Key to insert.
            current_time (float): Current time.
            current_idx (int): Index of the current request.
            testing_set (pd.DataFrame): Testing dataset for sequence extraction.
            pipeline_config (PipelineConfig): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If key insertion/eviction fails:
                * Key is unhashable or cache store/expiry dict
                  misconfigured (TypeError, AttributeError).
        """
        try:
            # Remove all expired keys
            # before insertion
            self._remove_expired_keys(current_time)

            # Check whether the cache is full
            if key not in self.store and len(self.store) >= self.maxsize:
                if current_idx >= self.model.seq_len:
                    # Convert dataset to DataFrame
                    testing_set = build_dataset(testing_set.data)

                    # Extract last (sequence length) accessed keys
                    last_accesses = (
                        testing_set.iloc[
                            current_idx - self.model.seq_len : current_idx
                        ][DATASET_COLUMN_REQUEST_NAME]
                        .values.astype(str)
                        .reshape(1, -1)
                    )

                    # Construct a DataFrame starting from
                    # extracted accesses
                    last_accesses_df = pd.DataFrame(
                        last_accesses,
                        columns=[
                            f"{DATASET_COLUMN_LR_PREVIOUS_REQUEST_PREFIX_NAME}{i}"
                            for i in range(1, self.model.seq_len + 1)
                        ],
                    )

                    # One-hot encode the extracted sequence
                    last_accesses_encoded = self.model.encoder.transform(
                        last_accesses_df,
                    ).toarray()

                    # Extract current features
                    current_features = (
                        testing_set[DATASET_PROCESSED_FEATURE_COLUMNS]
                        .iloc[[current_idx]]
                        .values
                    )

                    # Construct the final features as the combination
                    # of the encoded last accesses and the current features
                    X = np.concatenate(
                        [current_features, last_accesses_encoded],
                        axis=TENSOR_FEATURES_DIM,
                    )

                    # For each key, predict the probability
                    # of being used at the next step
                    probs = self.model.model.predict_proba(X)[LIST_FIRST_IDX]

                    # Key to evict as the one having the
                    # lowest probability among those in the cache
                    key_to_evict = min(
                        list(self.store.keys()),
                        key=lambda k: probs[
                            int(k) - pipeline_config.data.general.keys.min
                        ],
                    )
                else:
                    # No enough data available, random as fallback
                    key_to_evict = random.choice(list(self.store.keys()))

                # Evict key
                self.evict_key(key_to_evict)

                # Track eviction event
                self.metrics_logger.log_eviction(
                    key_to_evict,
                    current_time,
                )

            # Insert the key
            self._put_key(key, current_time)

        except (TypeError, AttributeError) as e:
            msg = "Key insertion/eviction to/from Logistic Regression cache failed"
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
                    "context": "LR Cache",
                },
            )
            raise RuntimeError(msg) from e
