from typing import Tuple

import numpy as np

from const import HOURS_IN_DAY
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def encode_time_trigonometrically(
    time_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a time list trigonometrically.

    This function converts a time list into
    two new features using sine and cosine transformations,
    allowing cyclical representation of time.

    Args:
        time_column (np.ndarray): Time list to be transformed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cosine and sine time produced.

    Raises:
        RuntimeError: If an error occurs while
                      encoding time trigonometrically, e.g.:
            * If the time list contains non-numeric values.
    """
    try:
        debug(
            f"(Time before normalization) Min:"
            f" {min(time_column)}, "
            f"Max: {max(time_column)}"
        )

        # Normalize time to [0, 2pi] so that
        # to have time in cycle
        time_in_cycle = (time_column % HOURS_IN_DAY) / HOURS_IN_DAY

        debug(
            f"(Time after normalization) Min:"
            f" {min(time_column)}, "
            f"Max: {max(time_column)}"
        )

        # Use normalized time in cycle
        # to get angles in radians
        angles = time_in_cycle * 2 * np.pi

        debug(f"Angles (radians) min: {angles.min()}, max: {angles.max()}")

        # Create sin and cos time
        sin_time, cos_time = np.sin(angles), np.cos(angles)

        info("Time encoded trigonometrically")

        return sin_time, cos_time
    except TypeError as e:
        msg = "Failed to encode time trigonometrically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
