from typing import Any, Optional

from mypy.checkexpr import defaultdict

from components.logs.levels.error_logger import error


def delete_item_from_cache(
    data: Any, key: Any, freq_data: Optional[defaultdict] = None
) -> None:
    """
    Delete a key and its associated item from the cache data.

    This function deletes the key from the provided cache data. If a
    frequency data structure is provided, it also removes the key from there.

    Args:
        data (Any): Cache data.
        key (Any): Key to delete.
        freq_data (Optional[defaultdict]): Optional frequency data structure to
                                           delete the key from.

    Returns:
        None

    Raises:
        RuntimeError: If deletion of the key from cache or frequency data fails:
            * The key does not exist in the cache data (KeyError).
            * The cache data structure is invalid or uninitialized (AttributeError).
    """
    try:
        # Delete the key from main cache
        del data[key]

        # Delete from frequency data if provided
        if freq_data is not None:
            freq_data.pop(key, None)
    except (AttributeError, KeyError) as e:
        msg = "Failed to delete item from cache"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
