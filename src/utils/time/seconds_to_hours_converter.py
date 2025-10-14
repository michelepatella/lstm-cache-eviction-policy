from typing import List

import numpy as np

from pipeline.const import SECONDS_IN_DAY, SECONDS_IN_HOUR
from utils.logs.levels.info_logger import info


def convert_seconds_to_hours(timestamps_seconds: List[float]) -> np.ndarray:
    """
    Convert timestamps from seconds to hours.

    This function converts provided timestamps in seconds
    to hours returning them.

    Args:
        timestamps_seconds (List[float]): Timestamps in seconds.

    Returns:
        np.ndarray: Converted timestamps in hours.
    """
    # Move from seconds to hours
    timestamps_hours = (
        np.array(timestamps_seconds) % SECONDS_IN_DAY
    ) / SECONDS_IN_HOUR

    info(f"Timestamps convertion from seconds to hours completed")

    return timestamps_hours
