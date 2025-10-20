from collections import OrderedDict
from typing import Any, Callable, Optional

from cachetools import Cache

from components.caches.implementations.items.evictions.oldest_item_evictor import (
    evict_oldest_item,
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


class FIFOCache(Cache):
    """
    FIFO (First-In-First-Out) cache implementation.

    Evicts the oldest inserted item when the maximum size is reached.

    Attributes:
        _data (OrderedDict): Internal data storage for cache items.
        callback (Optional[Callable]): Callback for evicted keys.
    """

    def __init__(
        self: "FIFOCache", maxsize: int, callback: Optional[Callable] = None
    ) -> None:
        """
        Initialize the FIFO cache.

        This function initializes the FIFO cache by setting up
        the data structure to collect data during simulations, and
        the optional callback provided.

        Args:
            self ("FIFOCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Optional[Callable]): Callback function invoked with the
                                           evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        # Fields initialization
        self._data = OrderedDict()
        self.callback = callback

        info(f"FIFO cache initialized (maxsize={self.maxsize})")

    def __getitem__(self: "FIFOCache", key: Any) -> Any:
        """
        Retrieve a key item from the FIFO cache.

        This function, given a key, retrieves its item from the FIFO cache.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.
        """
        debug(f"Key to get item from FIFO cache for: {key}")

        # Retrieve item
        item = get_item_from_cache(self._data, key)

        debug(f"FIFO cache item get: {item} (key={key})")

        return item

    def _evict_oldest_item(self: "FIFOCache") -> None:
        """
        Evict the oldest inserted item from the FIFO cache.

        This function removes the oldest key from the cache and triggers
        the callback if provided.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None
        """
        # Evict the oldest item from cache
        oldest_key, oldest_item = evict_oldest_item(self._data, self.callback)

        debug(f"FIFO cache item evicted: {oldest_item} (key={oldest_key})")

    def __setitem__(self: "FIFOCache", key: Any, item: Any) -> None:
        """
        Insert or update a key item in the FIFO cache.

        This function, given a key and its item, updates or inserts the
        provided item for the key in the FIFO cache (depending on whether
        the key is cached or not).

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        debug(f"Key and item to insert into FIFO cache: {key}, {item}")

        # Insert item into cache
        insert_item_into_cache(
            self._data, key, item, self.maxsize, self._evict_oldest_item
        )

        debug(f"FIFO cache item inserted: {item} (key={key})")

    def __delitem__(self: "FIFOCache", key: Any) -> None:
        """
        Delete a key and its item from the FIFO cache.

        This function, given a key, deletes it from the FIFO cache
        along with its item.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to delete.

        Returns:
            None
        """
        debug(f"Key to delete from FIFO cache: {key}")

        # Delete item from cache
        delete_item_from_cache(self._data, key)

        debug(f"FIFO cache key deleted: {key}")

    def __contains__(self: "FIFOCache", key: Any) -> bool:
        """
        Check if a key exists in the FIFO cache.

        This function, given a key, returns True if it exists in
        the FIFO cache, False otherwise.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to check.

        Returns:
            bool: True if key exists in the FIFO cache, False otherwise.
        """
        debug(f"Key existence check into FIFO cache: {key}")

        return check_item_into_cache(self._data, key)

    def pop(self: "FIFOCache", key: Any) -> Optional[Any]:
        """
        Remove a key from the FIFO cache and return its item.

        This function, given a key, removes it from the FIFO cache,
        returning its item. If the key is not in the cache, the function
        returns None.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to remove.

        Returns:
            Optional[Any]: Item associated with the key removed
                          (None if its key is not found into FIFO cache).
        """
        debug(f"Key to pop from FIFO cache: {key}")

        # Remove item from cache
        item = pop_item_from_cache(self._data, key)

        debug(f"FIFO cache item popped: {item} (key={key})")

    def __len__(self: "FIFOCache") -> int:
        """
        Get the number of items currently into FIFO cache.

        This function returns the number of items currently stored
        into FIFO cache.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        # Calculate cache size
        cache_size = calculate_cache_size(self._data)

        debug(f"FIFO cache size calculated: {cache_size}")

        return cache_size

    def clear(self: "FIFOCache") -> None:
        """
        Clear all items from the FIFO cache.

        This function clears the FIFO cache by removing all the items
        stored.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None
        """
        # Clear cache
        clear_cache(self._data)

        debug("FIFO cache cleared")
