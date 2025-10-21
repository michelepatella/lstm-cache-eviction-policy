import math

from components.const import TIME_HOURS_IN_DAY, TIME_SECONDS_IN_HOUR
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def decode_time_trigonometrically(
    sin_time: float,
    cos_time: float,
    cycle_length: float = TIME_HOURS_IN_DAY,
    cycle_unit_scale: float = TIME_SECONDS_IN_HOUR,
) -> float:
    """
    Decode time from its trigonometric encoding.

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
        debug(
            f"Sin time: {sin_time}, and cos time: {cos_time} to decode trigonometrically"
        )
        debug(f"Cycle length for trigonometric time decoding: {cycle_length}")
        debug(
            f"Cycle unit scale for trigonometric time decoding: {cycle_unit_scale}"
        )

        # Compute angle from sin and cos
        angle = math.atan2(sin_time, cos_time)

        # Normalize to positive range [0, 2pi]
        if angle < 0:
            angle += 2 * math.pi

        debug(f"Angle from sin and cos time: {angle} (after normalization)")

        # Convert angle back to specified unit
        current_time = angle / (2 * math.pi) * cycle_length * cycle_unit_scale

        debug(f"Time decoded trigonometrically: {current_time}")

        return current_time
    except TypeError as e:
        msg = "Failed to decode time trigonometrically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
