from datetime import timedelta

import numpy as np

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


def timestamps_seconds_to_hours(timestamps_seconds: list[float]) -> np.ndarray:
    """
    Convert timestamps from seconds to hours.

    This function converts timestamps from
    seconds to hours.

    Parameters:
        timestamps_seconds (list[float]): List of timestamps in seconds to be
                                          converted.

    Returns:
        np.ndarray: List of timestamps in hours just converted.

    Raises:
        TypeError: If the received timestamps in seconds list
                   is not a list of numbers.
    """
    debug(
        f"Tot. amount of timestamps in seconds to be converted: {len(timestamps_seconds)}"
    )

    try:
        # Get the total amount of seconds in hour
        seconds_in_hours = timedelta(hours=1).total_seconds()

        # Move from timestamps in seconds to hours, diving
        # by total amount of seconds in hour
        timestamps_hours = np.array(timestamps_seconds) / seconds_in_hours
    except TypeError as e:
        msg = f"Failed to convert timestamps from seconds to hours"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    debug(f"Tot. amount of resulting timestamps in hours: {len(timestamps_hours)}")

    info("Timestamps converted from seconds to hours")

    return timestamps_hours
