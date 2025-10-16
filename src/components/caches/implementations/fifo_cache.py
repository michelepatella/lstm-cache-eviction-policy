from collections import OrderedDict
from typing import Any, Callable

from cachetools import Cache

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


class FIFOCache(Cache):
    """
    FIFO (First-In-First-Out) cache implementation.

    Evicts the oldest inserted item when the maximum size is reached.

    Attributes:
        _data (OrderedDict): Internal data storage for cache items.
        callback (Callable | None): Callback for evicted keys.
    """

    def __init__(
        self: "FIFOCache", maxsize: int, callback: Callable = None
    ) -> None:
        """
        Initialize the FIFO cache.

        This function initializes the FIFO cache by setting up
        the data structure to collect data during simulations, and
        the optional callback provided.

        Args:
            self ("FIFOCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Callable): Callback function invoked
                                 with the evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        debug(f"FIFO cache max size: {self.maxsize}")

        # Fields initialization
        self._data = OrderedDict()
        self.callback = callback

        info("FIFO cache initialized")

    def __getitem__(self: "FIFOCache", key: int) -> Any:
        """
        Retrieve a key item from the FIFO cache.

        This function, given a key, retrieves its
        item from the FIFO cache.

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.

        Raises:
            RuntimeError: If an error occurs while retrieving
                          an item from the FIFO cache e.g.:
                            * The key does not exist in the
                              FIFO cache.
        """
        try:
            debug(f"Key to retrieve item from FIFO cache for: {key}")

            item = self._data[key]

            debug(f"FIFO cache item retrieved: {item}, for key: {key}")

            return item
        except KeyError as e:
            msg = "Failed to retrieve item from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def _evict_oldest_item(self: "FIFOCache") -> None:
        """
        Evict the oldest inserted item from the FIFO cache.

        This function removes the oldest key from the cache
        and triggers the callback if provided.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None
        """
        # Remove the oldest key and its item
        # from the cache
        oldest_key, oldest_item = self._data.popitem(last=False)

        debug(f"FIFO cache item evicted: {oldest_item}, for key: {oldest_key}")

        # Callback (if provided)
        if self.callback:
            self.callback(oldest_key)

    def __setitem__(self: "FIFOCache", key: int, item: Any) -> None:
        """
        Insert or update a key item in the FIFO cache.

        This function, given a key and its item, updates
        or inserts the provided item for the key in the
        FIFO cache (depending on whether the key is cached or not).

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        # If the key is not cached but
        # there is no space enough to
        # cache it
        if len(self._data) >= self.maxsize and key not in self._data:
            # Remove the oldest inserted
            # key from the cache, along with
            # its item
            self._evict_oldest_item()

        # Put the requested key in the cache,
        # along with its item
        self._data[key] = item

        debug(f"FIFO cache item inserted: {item}, for key: {key}")

    def __delitem__(self: "FIFOCache", key: int) -> None:
        """
        Delete a key and its item from the
        FIFO cache.

        This function, given a key, deletes it
        from the FIFO cache along with its item.

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to delete.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs while deleting
                          a key from the FIFO cache e.g.:
                            * The key does not exist in the
                              FIFO cache.
        """
        try:
            debug(f"Key to delete from FIFO cache: {key}")

            del self._data[key]
        except KeyError as e:
            msg = "Failed to delete key from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        debug(f"FIFO cache key deleted: {key}")

    def __contains__(self: "FIFOCache", key: int) -> bool:
        """
        Check if a key exists in the FIFO cache.

        This function, given a key, returns True if
        it exists in the FIFO cache, False otherwise.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to check.

        Returns:
            bool: True if key exists in the FIFO cache,
                  False otherwise.
        """
        debug(f"Key existence check in the FIFO cache: {key}")

        return key in self._data

    def pop(self: "FIFOCache", key: int) -> Any | None:
        """
        Remove a key from the FIFO cache and
        return its item.

        This function, given a key, removes it
        from the FIFO cache, returning its item.
        If the key is not in the cache, the function
        returns None instead of its item.

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to remove.

        Returns:
            Any | None: Item associated with the key removed
                        (None if its key is not found in the
                        FIFO cache)
        """
        # Remove key from the cache and
        # get its item (None if the key
        # is not found)
        item = self._data.pop(key, None)

        if item is not None:
            debug(f"FIFO cache item popped: {item}, for key: {key}")
        else:
            debug(f"FIFO cache pop attempted for non-existent key: {key}")

        return item

    def __len__(self: "FIFOCache") -> int:
        """
        Get the number of items currently
        in the FIFO cache.

        This function returns the number of items
        currently stored in the FIFO cache.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        return len(self._data)

    def clear(self: "FIFOCache") -> None:
        """
        Clear all items from the FIFO cache.

        This function clears the FIFO cache by
        removing all the items stored.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None
        """
        self._data.clear()
