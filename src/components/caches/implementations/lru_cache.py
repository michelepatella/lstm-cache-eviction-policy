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

        debug(
            "Cache initialization executed",
            extra={
                "maxsize": self.maxsize,
                "context": "LRU cache",
            },
        )

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
        # Retrieve item
        item = get_item_from_cache(self._data, key)

        try:
            # Reinsert key item to
            # mark it as recently used
            self._data[key] = item
        except (KeyError, TypeError, AttributeError) as e:
            msg = "Reinserting key into LRU cache failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "key": key,
                    "item_type": type(item).__name__,
                    "item_repr_len": (
                        len(repr(item)) if item is not None else 0
                    ),
                    "cache_type": type(self._data).__name__,
                    "cache_size": (
                        len(self._data)
                        if hasattr(self._data, "__len__")
                        else None
                    ),
                    "cache_name": getattr(self, "name", None),
                    "cache_capacity": getattr(self, "capacity", None),
                    "context": "LRU cache",
                },
            )
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
        evict_oldest_item(self._data, self.callback)

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
        # Delete item from cache
        delete_item_from_cache(self._data, key)

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
        # Remove item from cache
        return pop_item_from_cache(self._data, key)

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
        return calculate_cache_size(self._data)

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
