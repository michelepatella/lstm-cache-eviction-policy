from typing import Any

from components.logs.levels.error_logger import error


def clear_cache(
    data: Any,
    data_freq: Any | None = None,
) -> None:
    """Clear cache data.

    This function clears all items from the main cache data. If provided,
    it also clears the item frequency dictionary.

    Args:
        data (Any): Cache data structure to clear.
        data_freq (Any | None): Frequency dictionary to clear.

    Returns:
        None

    Raises:
        RuntimeError: If clearing cache fails:
            * The data structure is uninitialized or invalid
              (AttributeError, TypeError).
    """
    try:
        # Clear cache
        data.clear()

        # Clear frequencies if provided
        if data_freq is not None:
            data_freq.clear()
    except (AttributeError, TypeError) as e:
        msg = "Cache cleaning failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "data_type": type(data).__name__,
                "data_size": len(data) if hasattr(data, "__len__") else None,
                "data_freq_type": (
                    type(data_freq).__name__ if data_freq else None
                ),
                "data_freq_len": (
                    len(data_freq)
                    if data_freq and hasattr(data_freq, "__len__")
                    else None
                ),
                "context": "Cache cleaning",
            },
        )
        raise RuntimeError(msg) from e
