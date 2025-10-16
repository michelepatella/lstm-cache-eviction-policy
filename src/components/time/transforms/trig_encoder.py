from typing import Tuple

import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import HOURS_IN_DAY


def encode_time_trigonometrically(
    time_column: np.ndarray,
    cycle_length: float = HOURS_IN_DAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a time list trigonometrically.

    This function converts a time list into sine and cosine
    features, representing cyclical time.

    Args:
        time_column (np.ndarray): Time list to be transformed.
        cycle_length (float): Length of the repeating cycle (default 24).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - cos_time: Numpy array of cosine values representing cyclical time.
            - sin_time: Numpy array of sine values representing cyclical time.

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

        # Normalize time so that to have time in cycle
        time_in_cycle = (time_column % cycle_length) / cycle_length

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
