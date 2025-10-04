from collections import OrderedDict
from typing import Any, Callable

from cachetools import Cache

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


class LRUCache(Cache):
    """
    LRU (Least Recently Used) cache implementation.

    This cache evicts the least recently used item when the
    maximum size is reached. Optionally, a callback can be
    provided to handle evicted keys.
    """

    def __init__(
        self: "LRUCache", maxsize: int, callback: Callable = None
    ) -> None:
        """
        Initialize the LRU cache.

        This function initializes the LRU cache by setting up
        the data structure to collect data during simulation,
        and the optional callback provided.

        Parameters:
            self ("LRUCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Callable): Callback function invoked
                                 with the evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        debug(f"LRU cache max size: {self.maxsize}")

        # Fields initialization
        self._data: OrderedDict[int, Any] = OrderedDict()
        self.callback = callback

        info("LRU cache initialized")

    def __getitem__(self: "LRUCache", key: int) -> Any:
        """
        Retrieve a key item from the LRU cache.

        This function, given a key, retrieves its
        item from the LRU cache and moves it to the
        end of the internal ordered dictionary to
        mark it as recently used.

        Parameters:
            self ("LRUCache"): Current class instance.
            key (int): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.

        Raises:
            RuntimeError: If the key does not exist in the LRU cache.
        """
        try:
            debug(f"Key to retrieve item from LRU cache for: {key}")

            # Remove and reinsert key item
            # to mark it as recently used
            item = self._data.pop(key)
            self._data[key] = item

            debug(f"LRU cache item retrieved: {item}, for key: {key}")

            return item
        except KeyError as e:
            msg = "Failed to retrieve item from LRU cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __setitem__(self: "LRUCache", key: int, item: Any) -> None:
        """
        Insert or update a key item in the LRU cache.

        This function, given a key and its item, updates
        or inserts the provided item for the key in the
        LRU cache (depending on whether the key is already cached).

        Parameters:
            self ("LRUCache"): Current class instance.
            key (int): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        # If the key already exists,
        # move it to the end (recently used)
        if key in self._data:
            # Remove key item from the cache
            self._data.pop(key)

            debug(f"LRU cache item updated: {item}, for key: {key}")
        else:
            # Check whether there is no space
            # enough into the cache to store the key item
            if len(self._data) >= self.maxsize:
                # Remove the oldest item along with
                # its key from the cache, as the last
                # recently used
                oldest_key, oldest_item = self._data.popitem(last=False)

                debug(
                    f"LRU cache item evicted: {oldest_item}, "
                    f"for key: {oldest_key}"
                )

                # Callback (if provided)
                if self.callback:
                    self.callback(oldest_key)

            debug(f"LRU cache item inserted: {item}, for key: {key}")

        # Insert key as most recently used
        self._data[key] = item

    def __delitem__(self: "LRUCache", key: int) -> None:
        """
        Delete a key and its item from the LRU cache.

        This function, given a key, deletes it
        from the LRU cache.

        Parameters:
            self ("LRUCache"): Current class instance.
            key (int): Key to delete.

        Returns:
            None

        Raises:
            RuntimeError: If the key does not exist in the LRU cache.
        """
        try:
            debug(f"Key to delete from LRU cache: {key}")

            del self._data[key]
        except KeyError as e:
            msg = "Failed to delete key from LRU cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        debug(f"LRU cache key deleted: {key}")

    def __contains__(self: "LRUCache", key: int) -> bool:
        """
        Check if a key exists in the LRU cache.

        This function, given a key, returns True if
        it exists in the LRU cache, False otherwise.

        Parameters:
            self ("LRUCache"): Current class instance.
            key (int): Key to check.

        Returns:
            bool: True if key exists in the LRU cache,
                  False otherwise.
        """
        debug(f"Key existence check in LRU cache: {key}")

        return key in self._data

    def pop(self: "LRUCache", key: int) -> Any | None:
        """
        Remove a key from the LRU cache and return its item.

        This function, given a key, removes it
        from the LRU cache and returns its item.
        If the key is not found, it returns None instead.

        Parameters:
            self ("LRUCache"): Current class instance.
            key (int): Key to remove.

        Returns:
            Any | None: Item associated with the key removed
                        (None if its key is not found in the LRU cache).
        """
        # Pop key from cache
        item = self._data.pop(key, None)

        if item is not None:
            debug(f"LRU cache item popped: {item}, for key: {key}")
        else:
            debug(f"LRU cache pop attempted for non-existent key: {key}")

        return item

    def __len__(self: "LRUCache") -> int:
        """
        Get the number of items currently
        in the LRU cache.

        This function returns the number of items
        currently stored in the LRU cache.

        Parameters:
            self ("LRUCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        return len(self._data)

    def clear(self: "LRUCache") -> None:
        """
        Clear all items from the LRU cache.

        This function clears the LRU cache by
        removing all the items stored.

        Parameters:
            self ("LRUCache"): Current class instance.

        Returns:
            None
        """
        # Remove all data from cache
        self._data.clear()

        debug("LRU cache cleared")
