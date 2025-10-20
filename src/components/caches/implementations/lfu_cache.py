from collections import defaultdict
from typing import Any, Callable, Optional

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
from components.logs.levels.info_logger import info


class LFUCache(Cache):
    """
    LFU (Least Frequently Used) cache implementation.

    Evicts the least frequently used item when the maximum size is reached.

    Attributes:
        _data (dict): Internal data storage for cache items.
        _freq (defaultdict): Stores access frequency for each key.
        callback (Optional[Callable): Callback for evicted keys.
    """

    def __init__(
        self: "LFUCache", maxsize: int, callback: Optional[Callable] = None
    ) -> None:
        """
        Initialize the LFU cache.

        This function initializes the LFU cache by setting up the data
        structures to collect data during simulations, and the optional
        callback provided.

        Args:
            self ("LFUCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Optional[Callable]): Callback function invoked with the
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

        info(f"LFU cache initialized (maxsize={self.maxsize})")

    def __getitem__(self: "LFUCache", key: Any) -> Any:
        """
        Retrieve a key item from the LFU cache.

        This function, given a key, retrieves its item from the LFU
        cache and increments its access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (Any): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.
        """
        debug(f"Key to get item from LFU cache for: {key}")

        # Retrieve item
        item = get_item_from_cache(self._data, key)

        debug(f"LFU cache item get: {item} (key={key})")

        return item

    def _evict_least_frequent(self: "LFUCache") -> None:
        """
        Evict the least frequently used key.

        This function identifies the least frequently used key into the cache
        to be evicted. The tiebreak strategy implemented by the function
        consist of evicting the oldest key among candidate ones.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            None
        """
        # Evict the least frequent item from cache
        key_to_evict, min_freq = evict_least_frequent_item(
            self._data, self._freq, self.callback
        )

        debug(f"LFU cache key evicted: {key_to_evict} (frequency={min_freq})")

    def __setitem__(self: "LFUCache", key: Any, item: Any) -> None:
        """
        Insert or update a key item in the LFU cache.

        This function, given a key and its item, updates or inserts the
        provided item for the key in the LFU cache (depending on whether
        the key is already cached).

        Args:
            self ("LFUCache"): Current class instance.
            key (Any): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        debug(f"Key and item to insert into LFU cache: {key}, {item}")

        # Insert item into cache
        insert_item_into_cache(
            self._data,
            key,
            item,
            self.maxsize,
            self._evict_least_frequent,
            post_insert_callback=lambda k: self._freq.__setitem__(
                k, self._freq.get(k, 0) + 1
            ),
        )

        debug(f"LFU cache item inserted: {item} (key={key})")

    def __delitem__(self: "LFUCache", key: Any) -> None:
        """
        Delete a key and its item from the LFU cache.

        This function, given a key, deletes it from the LFU cache along
        with its access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (Any): Key to delete.

        Returns:
            None
        """
        debug(f"Key to delete from LFU cache: {key}")

        # Delete both item from cache and its frequency
        delete_item_from_cache(self._data, key, self._freq)

        debug(f"LFU cache key (and its frequency) deleted: {key}")

    def __contains__(self: "LFUCache", key: Any) -> bool:
        """
        Check if a key exists in the LFU cache.

        This function, given a key, returns True if it exists in the LFU
        cache, False otherwise.

        Args:
            self ("LFUCache"): Current class instance.
            key (Any): Key to check.

        Returns:
            bool: True if key exists in the LFU cache, False otherwise.
        """
        debug(f"Key existence check into LFU cache: {key}")

        return check_item_into_cache(self._data, key)

    def pop(self: "LFUCache", key: Any) -> Optional[Any]:
        """
        Remove a key from the LFU cache and return its item.

        This function, given a key, removes it from the LFU cache and
        returns its item. If the key is not found, it returns None.

        Args:
            self ("LFUCache"): Current class instance.
            key (Any): Key to remove.

        Returns:
            Optional[Any]: Item associated with the key removed
                           (None if its key is not found in the LFU cache).
        """
        debug(f"Key to pop from LFU cache: {key}")

        # Remove item from cache
        item = pop_item_from_cache(self._data, key, self._freq)

        debug(f"LFU cache item (and its frequency) popped: {item} (key={key})")

    def __len__(self: "LFUCache") -> int:
        """
        Get the number of items currently in the LFU cache.

        This function returns the number of items currently stored in
        the LFU cache.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        # Calculate cache size
        cache_size = calculate_cache_size(self._data)

        debug(f"LFU cache size calculated: {cache_size}")

        return cache_size

    def clear(self: "LFUCache") -> None:
        """
        Clear all items from the LFU cache.

        This function clears the LFU cache by
        removing all stored items and their frequencies.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            None
        """
        # Clear cache
        clear_cache(self._data, self._freq)

        debug("LFU cache cleared")
