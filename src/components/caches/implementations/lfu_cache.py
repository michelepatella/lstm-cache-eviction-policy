"""lfu_cache.py

Module implementing an LFU (Least Frequently Used) cache.

This module provides the `LFUCache` class, which manages key-value pairs in
a cache, evicting the least frequently used items when the maximum cache size
is reached.

Classes:
    LFUCache(maxsize, callback)
        LFU cache implementation supporting get, set, delete, pop, and clear
        operations with frequency tracking and optional eviction callbacks.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from cachetools import Cache

from components.caches.implementations.items.evictions.least_frequent_item_evictor import (
    evict_least_frequent_item,
)
from components.caches.implementations.items.operations.checker import (
    check_item_into_cache,
)
from components.caches.implementations.items.operations.deleter import (
    delete_item_from_cache,
)
from components.caches.implementations.items.operations.inserter import (
    insert_item_into_cache,
)
from components.caches.implementations.items.operations.popper import (
    pop_item_from_cache,
)
from components.caches.implementations.items.operations.retriever import (
    get_item_from_cache,
)
from components.caches.implementations.utils.cache_cleaner import clear_cache
from components.caches.implementations.utils.cache_size_calculator import (
    calculate_cache_size,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


class LFUCache(Cache):
    """LFU (Least Frequently Used) cache implementation.

    Evicts the least frequently used item when the maximum size is reached.

    Attributes:
        _data (dict): Internal data storage for cache items.
        _freq (defaultdict): Stores access frequency for each key.
        callback (Optional[Callable): Callback for evicted keys.
    """

    def __init__(
        self: "LFUCache",
        maxsize: int,
        callback: Callable | None = None,
    ) -> None:
        """Initialize the LFU cache.

        This function initializes the LFU cache by setting up the data
        structures to collect data during simulations, and the optional
        callback provided.

        Args:
            self ("LFUCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Callable | None): Callback function invoked with the
                                        evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        # Fields initialization
        self._data = {}
        self._freq = defaultdict()
        self.callback = callback

        debug(
            "Cache initialization executed",
            extra={
                "maxsize": self.maxsize,
                "context": "LFU cache",
            },
        )

    def __getitem__(self: "LFUCache", key: int) -> Any:
        """Retrieve a key item from the LFU cache.

        This function, given a key, retrieves its item from the LFU
        cache and increments its access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.

        Raises:
            RuntimeError: If getting the LFU cache item fails:
                * The requested key is not found in the LFU cache
                  (KeyError).
                * The cache data structure is invalid or uninitialized
                  (AttributeError).
                * The key is not hashable or the frequency update fails
                  (TypeError).
        """
        # Retrieve item
        item = get_item_from_cache(self._data, key)

        try:
            # Update item frequency
            self._freq[key] = self._freq.get(key, 0) + 1
        except (TypeError, KeyError) as e:
            msg = "LFU cache frequency update failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "item_type": type(item).__name__,
                    "freq_type": type(self._freq).__name__
                    if hasattr(self, "_freq")
                    else None,
                    "freq_len": len(self._freq)
                    if hasattr(self, "_freq") and self._freq
                    else 0,
                    "cache_type": type(self._data).__name__,
                    "cache_size": len(self._data)
                    if hasattr(self._data, "__len__")
                    else None,
                    "context": "LFU cache",
                },
            )
            raise RuntimeError(msg) from e

        return item

    def _evict_least_frequent(self: "LFUCache") -> None:
        """Evict the least frequently used key.

        This function identifies the least frequently used key into the cache
        to be evicted. The tiebreak strategy implemented by the function
        consist of evicting the oldest key among candidate ones.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            None
        """
        # Evict the least frequent item from cache
        evict_least_frequent_item(self._data, self._freq, self.callback)

    def __setitem__(self: "LFUCache", key: int, item: Any) -> None:
        """Insert or update a key item in the LFU cache.

        This function, given a key and its item, updates or inserts the
        provided item for the key in the LFU cache (depending on whether
        the key is already cached).

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        # Insert item into cache
        insert_item_into_cache(
            self._data,
            key,
            item,
            self.maxsize,
            self._evict_least_frequent,
            post_insert_callback=lambda k: self._freq.__setitem__(
                k,
                self._freq.get(k, 0) + 1,
            ),
        )

    def __delitem__(self: "LFUCache", key: int) -> None:
        """Delete a key and its item from the LFU cache.

        This function, given a key, deletes it from the LFU cache along
        with its access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to delete.

        Returns:
            None
        """
        # Delete both item from cache and its frequency
        delete_item_from_cache(self._data, key, self._freq)

    def __contains__(self: "LFUCache", key: int) -> bool:
        """Check if a key exists in the LFU cache.

        This function, given a key, returns True if it exists in the LFU
        cache, False otherwise.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to check.

        Returns:
            bool: True if key exists in the LFU cache, False otherwise.
        """
        return check_item_into_cache(self._data, key)

    def pop(self: "LFUCache", key: int) -> Any | None:
        """Remove a key from the LFU cache and return its item.

        This function, given a key, removes it from the LFU cache and
        returns its item. If the key is not found, it returns None.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to remove.

        Returns:
            Any | None: Item associated with the key removed
                        (None if its key is not found in the LFU cache).
        """
        # Remove item from cache
        return pop_item_from_cache(self._data, key, self._freq)

    def __len__(self: "LFUCache") -> int:
        """Get the number of items currently in the LFU cache.

        This function returns the number of items currently stored in
        the LFU cache.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        # Calculate cache size
        return calculate_cache_size(self._data)

    def clear(self: "LFUCache") -> None:
        """Clear all items from the LFU cache.

        This function clears the LFU cache by
        removing all stored items and their frequencies.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            None
        """
        # Clear cache
        clear_cache(self._data, self._freq)
