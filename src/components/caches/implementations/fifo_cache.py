from collections import OrderedDict
from typing import Any, Callable, Optional

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
            RuntimeError: If retrieving the FIFO cache item fails:
                * The requested key is not found in the FIFO cache (KeyError).
                * Internal cache data structure is invalid or uninitialized
                  (AttributeError, TypeError).
        """
        try:
            debug(f"Key to retrieve item from FIFO cache for: {key}")

            item = self._data[key]

            info(f"FIFO cache item retrieved: {item} (key={key})")

            return item
        except KeyError as e:
            msg = "Failed to retrieve item from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def _evict_oldest_item(self: "FIFOCache") -> None:
        """
        Evict the oldest inserted item from the FIFO cache.

        This function removes the oldest key from the cache and triggers
        the callback if provided.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None

        Raises:
            RuntimeError: If eviction of the oldest FIFO cache item fails:
                * The cache is empty when attempting to evict an item (KeyError).
                * The internal cache data structure is invalid or uninitialized
                  (AttributeError, TypeError).
        """
        try:
            # Remove the oldest key and its item
            # from the cache
            oldest_key, oldest_item = self._data.popitem(last=False)

            info(f"FIFO cache item evicted: {oldest_item} (key={oldest_key})")

            # Callback (if provided)
            if self.callback:
                self.callback(oldest_key)
        except (KeyError, AttributeError, TypeError) as e:
            msg = "Failed to evict item from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __setitem__(self: "FIFOCache", key: int, item: Any) -> None:
        """
        Insert or update a key item in the FIFO cache.

        This function, given a key and its item, updates or inserts the
        provided item for the key in the FIFO cache (depending on whether
        the key is cached or not).

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None

        Raises:
            RuntimeError: If inserting the item into the FIFO cache fails:
                * The key is not hashable or the cache dictionary does not support
                  assignment (TypeError).
                * The cache dictionary is not initialized or invalid (AttributeError).
                * Evicting the oldest item fails due to the cache being empty or
                  internal errors (RuntimeError).
        """
        try:
            debug(f"Key and item to insert into FIFO cache: {key}, {item}")

            # If the key is not cached but there is
            # no space enough to cache it
            if len(self._data) >= self.maxsize and key not in self._data:
                # Remove the oldest inserted key from
                # the cache, along with its item
                self._evict_oldest_item()

            # Put the requested key in the cache,
            # along with its item
            self._data[key] = item

            info(f"FIFO cache item inserted: {item} (key={key})")
        except (TypeError, AttributeError, RuntimeError) as e:
            msg = "Failed to insert item into FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __delitem__(self: "FIFOCache", key: int) -> None:
        """
        Delete a key and its item from the FIFO cache.

        This function, given a key, deletes it from the FIFO cache
        along with its item.

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to delete.

        Returns:
            None

        Raises:
            RuntimeError: If deleting the key from the FIFO cache fails:
                * The key does not exist in the cache (KeyError).
                * The cache dictionary is not initialized or invalid
                  (AttributeError).
        """
        try:
            debug(f"Key to delete from FIFO cache: {key}")

            del self._data[key]

            info(f"FIFO cache key deleted: {key}")
        except KeyError as e:
            msg = "Failed to delete key from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __contains__(self: "FIFOCache", key: int) -> bool:
        """
        Check if a key exists in the FIFO cache.

        This function, given a key, returns True if it exists in
        the FIFO cache, False otherwise.

        Args:
            self ("FIFOCache"): Current class instance.
            key (Any): Key to check.

        Returns:
            bool: True if key exists in the FIFO cache, False otherwise.

        Raises:
            RuntimeError: If checking key existence fails:
                * The cache data structure is uninitialized or invalid
                  (AttributeError).
                * The key is not hashable and cannot be used in the cache
                  (TypeError).
        """
        try:
            debug(f"Key existence check into FIFO cache: {key}")

            return key in self._data
        except (AttributeError, TypeError) as e:
            msg = "Failed to check key existence into FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def pop(self: "FIFOCache", key: int) -> Optional[Any]:
        """
        Remove a key from the FIFO cache and return its item.

        This function, given a key, removes it from the FIFO cache,
        returning its item. If the key is not in the cache, the function
        returns None.

        Args:
            self ("FIFOCache"): Current class instance.
            key (int): Key to remove.

        Returns:
            Optional[Any]: Item associated with the key removed
                          (None if its key is not found into FIFO cache).

        Raises:
            RuntimeError: If popping the key from FIFO cache fails:
                * The cache data structure is uninitialized or invalid
                  (AttributeError).
                * The key is not hashable and cannot be used in the cache
                  (TypeError).
        """
        try:
            # Remove key from the cache and get its item
            # (None if the key is not found)
            item = self._data.pop(key, None)

            info(f"FIFO cache item popped: {item} (key={key})")

            return item
        except (AttributeError, TypeError) as e:
            msg = "Failed to pop item from FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __len__(self: "FIFOCache") -> int:
        """
        Get the number of items currently into FIFO cache.

        This function returns the number of items currently stored
        into FIFO cache.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            int: Number of cached items.

        Raises:
            RuntimeError: If accessing the cache size fails:
                * The cache data structure is uninitialized or invalid
                  (AttributeError).
                * The cache data structure does not support the method
                  to calculate its length (TypeError).
        """
        try:
            # Calculate cache size
            cache_size = len(self._data)

            info(f"FIFO cache size calculated: {cache_size}")

            return cache_size
        except (AttributeError, TypeError) as e:
            msg = "Failed to get length of FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def clear(self: "FIFOCache") -> None:
        """
        Clear all items from the FIFO cache.

        This function clears the FIFO cache by removing all the items
        stored.

        Args:
            self ("FIFOCache"): Current class instance.

        Returns:
            None

        Raises:
            RuntimeError: If clearing the FIFO cache fails:
                * The cache data structure is uninitialized or invalid
                  (AttributeError).
                * The cache data structure does not support the method
                  to clear the cache (TypeError).
        """
        try:
            self._data.clear()
            info("FIFO cache cleared")
        except (AttributeError, TypeError) as e:
            msg = "Failed to clear FIFO cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
