from collections import defaultdict
from typing import Any, Callable

from cachetools import Cache

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


class LFUCache(Cache):
    """
    LFU (Least Frequently Used) cache implementation.

    Evicts the least frequently used item when the maximum size is reached.

    Attributes:
        _data (dict): Internal data storage for cache items.
        _freq (defaultdict): Stores access frequency for each key.
        callback (Callable | None): Optional callback for evicted keys.
    """

    def __init__(
        self: "LFUCache", maxsize: int, callback: Callable = None
    ) -> None:
        """
        Initialize the LFU cache.

        This function initializes the LFU cache by setting up
        the data structures to collect data during simulation,
        and the optional callback provided.

        Args:
            self ("LFUCache"): Current class instance.
            maxsize (int): Maximum size of the cache.
            callback (Callable): Callback function invoked
                                 with the evicted key.

        Returns:
            None
        """
        # Class instantiation
        super().__init__(maxsize)

        debug(f"LFU cache max size: {self.maxsize}")

        # Fields initialization
        self._data = {}
        self._freq = defaultdict()
        self.callback = callback

        info("LFU cache initialized")

    def _increment_frequency(self: "LFUCache", key: int) -> None:
        """
        Increment key access frequency.

        This function increments the frequency of a
        given key stored in the cache.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to increment frequency.

        Returns:
            None
        """
        # Increment key access by one
        self._freq[key] += 1

        debug(
            f"LFU cache frequency incremented for key: {key}, now: {self._freq[key]}"
        )

    def __getitem__(self: "LFUCache", key: int) -> Any:
        """
        Retrieve a key item from the LFU cache.

        This function, given a key, retrieves its
        item from the LFU cache and increments its
        access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to look up in the cache.

        Returns:
            Any: Retrieved key item.

        Raises:
            RuntimeError: If the key does not exist in the LFU cache.
        """
        try:
            debug(f"Key to retrieve item from LFU cache for: {key}")

            # Retrieve key item
            item = self._data[key]

            # Increment key item access
            # frequency by one
            self._increment_frequency(key)

            debug(
                f"LFU cache item retrieved: {item}, "
                f"for key: {key}, frequency updated "
                f"to: {self._freq[key]}"
            )

            return item
        except KeyError as e:
            msg = "Failed to retrieve item from LFU cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def _evict_least_frequent(self: "LFUCache") -> None:
        """
        Evict the least frequently used key.

        This function identifies the least frequently used
        key into the cache to be evicted. The tiebreak
        strategy implemented by the function consist of
        evicting the oldest key among candidate ones.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            None
        """
        # Identify least frequency
        min_freq = min(self._freq.values())

        # Identify candidate keys with minimum frequency
        evicted_candidates = [
            k for k, f in self._freq.items() if f == min_freq
        ]

        # Tiebreak strategy: select the oldest
        # key among them
        key_to_evict = evicted_candidates[0]

        # Remove selected key from cache
        # and frequency dictionary
        del self._data[key_to_evict]
        del self._freq[key_to_evict]

        debug(f"LFU cache key evicted: {key_to_evict}, frequency: {min_freq}")

        # Callback if present
        if self.callback:
            self.callback(key_to_evict)

    def __setitem__(self: "LFUCache", key: int, item: Any) -> None:
        """
        Insert or update a key item in the LFU cache.

        This function, given a key and its item, updates
        or inserts the provided item for the key in the
        LFU cache (depending on whether the key is already cached).

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to store in the cache.
            item (Any): Value associated with the key.

        Returns:
            None
        """
        # Check whether there is no space
        # enough into the cache to store the key item which
        # is not into the cache
        if len(self._data) >= self.maxsize and key not in self._data:
            # Evict the least frequently used
            # key from the cache
            self._evict_least_frequent()

        # Update key item and increment
        # its access frequency
        self._data[key] = item
        self._increment_frequency(key)

        debug(f"LFU cache item inserted: {item}, for key: {key}")

    def __delitem__(self: "LFUCache", key: int) -> None:
        """
        Delete a key and its item from the LFU cache.

        This function, given a key, deletes it
        from the LFU cache along with its access frequency.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to delete.

        Returns:
            None

        Raises:
            RuntimeError: If the key does not exist in the LFU cache.
        """
        try:
            debug(f"Key to delete from LFU cache: {key}")

            # Remove both key and its
            # access frequency
            del self._data[key]
            del self._freq[key]
        except KeyError as e:
            msg = "Failed to delete key from LFU cache"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        debug(f"LFU cache key deleted: {key}")

    def __contains__(self: "LFUCache", key: int) -> bool:
        """
        Check if a key exists in the LFU cache.

        This function, given a key, returns True if
        it exists in the LFU cache, False otherwise.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to check.

        Returns:
            bool: True if key exists in the LFU cache,
                  False otherwise.
        """
        debug(f"Key existence check in LFU cache: {key}")
        return key in self._data

    def pop(self: "LFUCache", key: int) -> Any | None:
        """
        Remove a key from the LFU cache and return its item.

        This function, given a key, removes it
        from the LFU cache and returns its item.
        If the key is not found, it returns None instead.

        Args:
            self ("LFUCache"): Current class instance.
            key (int): Key to remove.

        Returns:
            Any | None: Item associated with the key removed
                        (None if its key is not found in the LFU cache).
        """
        # Pop the key from the cache and
        # get its item
        item = self._data.pop(key, None)

        # If the key popped was in the
        # cache
        if item is not None:
            # Remove its access frequency too
            self._freq.pop(key, None)

            debug(f"LFU cache item popped: {item}, for key: {key}")
        else:
            debug(f"LFU cache pop attempted for non-existent key: {key}")

        return item

    def __len__(self: "LFUCache") -> int:
        """
        Get the number of items currently
        in the LFU cache.

        This function returns the number of items
        currently stored in the LFU cache.

        Args:
            self ("LFUCache"): Current class instance.

        Returns:
            int: Number of cached items.
        """
        return len(self._data)

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
        # Clear cache removing both
        # data and access frequencies
        self._data.clear()
        self._freq.clear()

        debug("LFU cache cleared")
