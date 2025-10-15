import math
from pipeline.const import HOURS_IN_DAY, SECONDS_IN_HOUR
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def decode_time_trigonometrically(
    sin_time: float,
    cos_time: float,
    cycle_length: float = HOURS_IN_DAY,
    cycle_unit_scale: float = SECONDS_IN_HOUR,
) -> float:
    """
    Decode time in seconds from its trigonometric encoding.

    This function converts sin and cos time, representing
    the trigonometric encoding of time, back to time
    representation in seconds.

    Args:
        sin_time (float): Sine component of encoded time.
        cos_time (float): Cosine component of encoded time.
        cycle_length (float): Length of one full cycle.
        cycle_unit_scale (float): Scaling factor for the output unit.

    Returns:
        float: Decoded time in seconds within the given cycle.

    Raises:
        RuntimeError: If an error occurs while decoding time
                      trigonometrically e.g.:
                      * Invalid input type.
    """
    try:
        # Compute angle from sin and cos
        angle = math.atan2(sin_time, cos_time)

        # Normalize to positive range [0, 2pi]
        if angle < 0:
            angle += 2 * math.pi

        # Convert angle back to seconds
        current_time = angle / (2 * math.pi) * cycle_length * cycle_unit_scale

        info("Time decoded trigonometrically")

        return current_time
    except TypeError as e:
        msg = "Failed to decode time trigonometrically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
