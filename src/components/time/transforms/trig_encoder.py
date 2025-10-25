from typing import Tuple

import numpy as np

from components.const import TIME_HOURS_IN_DAY
from components.logs.levels.error_logger import error


def encode_time_trigonometrically(
    timestamps: np.ndarray,
    cycle_length: float = TIME_HOURS_IN_DAY,
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
        # Normalize time so that to be in cycle
        time_in_cycle = (timestamps % cycle_length) / cycle_length

        # Compute angles (in radians) of
        # timestamps in cycle
        angles = time_in_cycle * 2 * np.pi

        # Get sin and cos components from angles
        sin_time = np.sin(angles)
        cos_time = np.cos(angles)

        return sin_time, cos_time
    except TypeError as e:
        msg = "Time trigonometric encoding failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "timestamps_shape": getattr(timestamps, "shape", None),
                "cycle_length": cycle_length,
                "context": "Time trigonometric encoding",
            },
        )
        raise RuntimeError(msg) from e
