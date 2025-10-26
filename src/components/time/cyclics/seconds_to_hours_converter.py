import numpy as np

from components.const import TIME_SECONDS_IN_DAY, TIME_SECONDS_IN_HOUR
from components.logs.levels.error_logger import error


def convert_seconds_to_hours_cyclic(
    timestamps_seconds: list[float],
) -> np.ndarray:
    """Convert timestamps from seconds to hours (cyclic over a day).

    This function converts provided timestamps in seconds to hours, wrapping
    them within a 24-hour day (0-24h).

    Args:
        timestamps_seconds (list[float]): Timestamps in seconds.

    Returns:
        np.ndarray: Converted timestamps in hours.

    Raises:
        RuntimeError: If cyclical conversion of timestamps from seconds
                      to hours fails:
            * Input timestamp list contains invalid (non-numeric) elements
              or types (TypeError).
            * Conversion of timestamp list to NumPy array or modular
              arithmetic fails due to invalid data (ValueError).
    """
    try:
        # Move from seconds to hours
        timestamps_hours = (
            np.array(timestamps_seconds) % TIME_SECONDS_IN_DAY
        ) / TIME_SECONDS_IN_HOUR

        return timestamps_hours
    except (TypeError, ValueError) as e:
        msg = "Seconds to hours cyclic conversion failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "input_type": type(timestamps_seconds).__name__,
                "input_length": (
                    len(timestamps_seconds)
                    if hasattr(timestamps_seconds, "__len__")
                    else None
                ),
                "context": "Seconds to hours cyclic conversion",
            },
        )
        raise RuntimeError(msg) from e
