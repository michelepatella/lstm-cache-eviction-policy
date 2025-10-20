from typing import Tuple

import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import HOURS_IN_DAY


def encode_time_trigonometrically(
    timestamps: np.ndarray,
    cycle_length: float = HOURS_IN_DAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a list of timestamps trigonometrically.

    This function encodes a list of timestamps trigonometrically through
    their sine and cosine representations.

    Args:
        timestamps (np.ndarray): Time list to be encoded.
        cycle_length (float): Length of the repeating cycle.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - cos_time: Numpy array of cosine values representing cyclical time.
            - sin_time: Numpy array of sine values representing cyclical time.
    Raises:
        RuntimeError: If encoding time trigonometrically fails:
            * Input timestamps are not numeric or not iterable (TypeError).
            * Cycle length is invalid or non-numeric (TypeError).
            * Computation of sine or cosine values fails due to invalid input
              (TypeError).
    """
    try:
        debug(
            f"(Timestamps to encode before normalization) min:"
            f" {min(timestamps)}, max: {max(timestamps)}"
        )

        # Normalize time so that to be in cycle
        time_in_cycle = (timestamps % cycle_length) / cycle_length
        debug(
            f"(Timestamps to encode after normalization) min:"
            f" {min(timestamps)}, max: {max(timestamps)}"
        )

        # Compute angles (in radians) of
        # timestamps in cycle
        angles = time_in_cycle * 2 * np.pi
        debug(
            f"Cyclic timestamp angles (radians) min: {angles.min()},"
            f" max: {angles.max()}"
        )

        # Get sin and cos components from angles
        sin_time = np.sin(angles)
        cos_time = np.cos(angles)

        info("Time encoded trigonometrically")

        return sin_time, cos_time
    except TypeError as e:
        msg = "Failed to encode time trigonometrically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
