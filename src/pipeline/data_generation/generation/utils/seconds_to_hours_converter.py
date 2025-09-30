from typing import List

import numpy as np

from const import SECONDS_IN_HOUR
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def seconds_to_hours(
    timestamps_seconds: List[float],
) -> np.ndarray:
    """
    Convert timestamps from seconds to hours.

    This function converts timestamps from
    seconds to hours.

    Parameters:
        timestamps_seconds (List[float]): List of timestamps in seconds to be
                                          converted.

    Returns:
        np.ndarray: List of timestamps in hours just converted.

    Raises:
        RuntimeError: If the received timestamps in seconds list
                      is not a list of numbers.
    """
    try:
        # Move from timestamps in seconds to hours, diving
        # by total amount of seconds in hour
        timestamps_hours = (
            np.array(timestamps_seconds, dtype=float) / SECONDS_IN_HOUR
        )
    except TypeError as e:
        msg = "Failed to convert timestamps from seconds to hours"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(
        f"{len(timestamps_seconds)} timestamps in seconds converted to {len(timestamps_hours)} timestamps in hours"
    )

    return timestamps_hours
