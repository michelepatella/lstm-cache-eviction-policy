"""trig_decoder.py

Utility module for decoding time from trigonometric representation.

This module provides the `decode_time_trigonometrically` function,
which converts sine and cosine components of encoded time back to a
numeric time value within a specified cycle length and unit scale.

Functions:
    decode_time_trigonometrically(
        sin_time: float,
        cos_time: float,
        cycle_length: float = TIME_HOURS_IN_DAY,
        cycle_unit_scale: float = TIME_SECONDS_IN_HOUR
    ) -> float
        Decodes trigonometric time encoding into a standard time value.
"""

import math

from components.const import TIME_HOURS_IN_DAY, TIME_SECONDS_IN_HOUR
from components.logs.levels.error_logger import error


def decode_time_trigonometrically(
    sin_time: float,
    cos_time: float,
    cycle_length: float = TIME_HOURS_IN_DAY,
    cycle_unit_scale: float = TIME_SECONDS_IN_HOUR,
) -> float:
    """Decode time from its trigonometric encoding.

    This function converts sin and cos time, representing the trigonometric
    encoding of time, back to time representation according to provided cycle
    length and unit scale.

    Args:
        sin_time (float): Sine component of encoded time.
        cos_time (float): Cosine component of encoded time.
        cycle_length (float): Length of one full cycle.
        cycle_unit_scale (float): Scaling factor for the output unit.

    Returns:
        float: Decoded time in specified unit within the given cycle.
    """
    try:
        # Compute angle from sin and cos
        angle = math.atan2(sin_time, cos_time)

        # Normalize to positive range [0, 2pi]
        if angle < 0:
            angle += 2 * math.pi

        # Convert angle back to specified unit
        current_time = angle / (2 * math.pi) * cycle_length * cycle_unit_scale

        return current_time
    except TypeError as e:
        msg = "Time trigonometric decoding failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "sin_time": sin_time,
                "cos_time": cos_time,
                "cycle_length": cycle_length,
                "cycle_unit_scale": cycle_unit_scale,
                "context": "Time trigonometric decoding",
            },
        )
        raise RuntimeError(msg) from e
