from typing import List

import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import SECONDS_IN_DAY, SECONDS_IN_HOUR


def convert_seconds_to_hours_cyclic(
    timestamps_seconds: List[float],
) -> np.ndarray:
    """
    Convert timestamps from seconds to hours (cyclic over a day).

    This function converts provided timestamps in seconds to hours, wrapping
    them within a 24-hour day (0-24h).

    Args:
        timestamps_seconds (List[float]): Timestamps in seconds.

    Returns:
        np.ndarray: Converted timestamps in hours.

    Raises:
        RuntimeError: If cyclical conversion of timestamps from seconds to hours fails:
            * Input timestamp list contains invalid (non-numeric) elements or types
              (TypeError).
            * Conversion of timestamp list to NumPy array or modular arithmetic fails
              due to invalid data (ValueError).
    """
    try:
        # Move from seconds to hours
        timestamps_hours = (
            np.array(timestamps_seconds) % SECONDS_IN_DAY
        ) / SECONDS_IN_HOUR

        debug("Timestamps converted from seconds to hours cyclically")

        return timestamps_hours
    except (TypeError, ValueError) as e:
        msg = "Failed to convert timestamps in seconds to hours cyclically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
