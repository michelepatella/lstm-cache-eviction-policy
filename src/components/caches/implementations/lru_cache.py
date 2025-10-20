from collections import OrderedDict
from typing import Any, Callable, Optional

from cachetools import Cache

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
from components.caches.implementations.items.evictions.oldest_item_evictor import (
    evict_oldest_item,
)
from components.caches.implementations.utils.cache_cleaner import clear_cache
from components.caches.implementations.utils.cache_size_calculator import (
    calculate_cache_size,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


class LRUCache(Cache):
    """
    LRU (Least Recently Used) cache implementation.

    Evicts the least recently used item when the maximum size is reached.

    Attributes:
        _data (OrderedDict): Internal data storage for cache items
                             in order of use.
        callback (Optional[Callable]): Callback for evicted keys.
    """

    def __init__(
        self: "LRUCache", maxsize: int, callback: Optional[Callable] = None
    ) -> None:
        """
        Initialize the LRU cache.

        This function initializes the LRU cache by setting up the data
        structure to collect data during simulations, and the optional
        callback provided.

        Args:
            self ("LRUCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Callable): Callback function invoked with the evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        # Fields initialization
        self._data = OrderedDict()
        self.callback = callback

        info(f"LRU cache initialized (maxsize={self.maxsize})")

    def __getitem__(self: "LRUCache", key: Any) -> Any:
        """
        Retrieve a key item from the LRU cache.

        This function, given a key, retrieves its item from the LRU
        cache and moves it to the end of the internal ordered dictionary to
        mark it as recently used.

        Args:
            self ("LRUCache"): Current class instance.
            key (Any): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.

        Raises:
            RuntimeError: If getting the LRU cache item fails:
                * The requested key is not found in the LRU cache (KeyError).
                * The cache data structure is invalid or uninitialized
                  (AttributeError).
                * The key is not hashable or the cache does not support
                  item assignment (TypeError).
        """
        debug(f"Key to get item from LRU cache for: {key}")

        # Retrieve item
        item = get_item_from_cache(self._data, key)

        debug(f"LRU cache item get: {item} (key={key})")

        try:
            # Reinsert key item to
            # mark it as recently used
            self._data[key] = item
        except (KeyError, TypeError, AttributeError) as e:
            msg = "Failed to reinsert key into LRU cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        return item

    def _evict_oldest_item(self: "LRUCache") -> None:
        """
        Evict the oldest item from the LRU cache.

        This function removes the least recently used key from the cache
        along with its item. If a callback is provided, it is invoked
        with the evicted key.

        Args:
            self ("LRUCache"): Current class instance.

        Returns:
            None
        """
        # Evict the oldest item from cache
        oldest_key, oldest_item = evict_oldest_item(self._data, self.callback)

        debug(f"LRU cache item evicted: {oldest_item} (key={oldest_key})")

    def __setitem__(self: "LRUCache", key: Any, item: Any) -> None:
        """
        Insert or update a key item in the LRU cache.

        This function, given a key and its item, updates or inserts
        the provided item for the key in the LRU cache (depending on
        whether the key is already cached).

        Args:
            self ("LRUCache"): Current class instance.
            key (Any): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        debug(f"Key and item to insert into LRU cache: {key}, {item}")

        # Insert item into cache
        insert_item_into_cache(
            self._data,
            key,
            item,
            self.maxsize,
            self._evict_oldest_item,
            pre_insert_callback=lambda k: (
                self._data.pop(k) if k in self._data else None
            ),
        )

        debug(f"LRU cache item inserted: {item} (key={key})")

    def __delitem__(self: "LRUCache", key: Any) -> None:
        """
        Delete a key and its item from the LRU cache.

        This function, given a key, deletes it from the LRU cache.

        Args:
            self ("LRUCache"): Current class instance.
            key (Any): Key to delete.

        Returns:
            None
        """
        debug(f"Key to delete from LRU cache: {key}")

        # Delete item from cache
        delete_item_from_cache(self._data, key)

        debug(f"LRU cache key deleted: {key}")

    def __contains__(self: "LRUCache", key: Any) -> bool:
        """
        Check if a key exists in the LRU cache.

        This function, given a key, returns True if
        it exists in the LRU cache, False otherwise.

        Args:
            self ("LRUCache"): Current class instance.
            key (Any): Key to check.

        Returns:
            bool: True if key exists in the LRU cache, False otherwise.
        """
        debug(f"Key existence check into LRU cache: {key}")

        return check_item_into_cache(self._data, key)

    def pop(self: "LRUCache", key: Any) -> Optional[Any]:
        """
        Remove a key from the LRU cache and return its item.

        This function, given a key, removes it from the LRU cache
        and returns its item. If the key is not found, it returns None instead.

        Args:
            self ("LRUCache"): Current class instance.
            key (Any): Key to remove.

        Returns:
            Optional[Any]: Item associated with the key removed
                          (None if its key is not found in the LRU cache).
        """
        debug(f"Key to pop from LRU cache: {key}")

        # Remove item from cache
        item = pop_item_from_cache(self._data, key)

        debug(f"LRU cache item popped: {item} (key={key})")

    def __len__(self: "LRUCache") -> int:
        """
        Get the number of items currently
        in the LRU cache.

        This function returns the number of items
        currently stored in the LRU cache.

        Args:
            self ("LRUCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        # Calculate cache size
        cache_size = calculate_cache_size(self._data)

        debug(f"LRU cache size calculated: {cache_size}")

        return cache_size

    def clear(self: "LRUCache") -> None:
        """
        Clear all items from the LRU cache.

        This function clears the LRU cache by
        removing all the items stored.

        Args:
            self ("LRUCache"): Current class instance.

        Returns:
            None
        """
        # Clear cache
        clear_cache(self._data)

        debug("LRU cache cleared")
