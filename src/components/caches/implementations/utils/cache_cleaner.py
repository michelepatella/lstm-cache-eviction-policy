from typing import Any, Optional

from components.logs.levels.error_logger import error


def clear_cache(
    data: Any,
    data_freq: Optional[Any] = None,
) -> None:
    """
    Clear cache data.

    This function clears all items from the main cache data. If provided,
    it also clears the item frequency dictionary.

    Args:
        data (Any): Cache data structure to clear.
        data_freq (Optional[Any]): Frequency dictionary to clear.

    Returns:
        None

    Raises:
        RuntimeError: If clearing cache fails:
            * The data structure is uninitialized or invalid (AttributeError, TypeError).
    """
    try:
        # Clear cache
        data.clear()

        # Clear frequencies if provided
        if data_freq is not None:
            data_freq.clear()
    except (AttributeError, TypeError) as e:
        msg = "Failed to clear cache"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
