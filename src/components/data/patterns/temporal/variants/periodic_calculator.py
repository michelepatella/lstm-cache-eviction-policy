import numpy as np

from components.const import TIME_HOURS_IN_DAY
from components.logs.levels.error_logger import error


def calculate_periodic_component(
    scale: int,
    amplitude: int,
    current_hour_in_day: float,
) -> float:
    """Calculate the periodic component given periodic scale and
    amplitude, as well as current hour in day.

    This function calculates a periodic component for current hour
    in day, given scale and amplitude, by adding the base scale to
    the cosine-modulated amplitude.

    Args:
        scale (int): Scale of periodic component to be calculated.
        amplitude (int): Amplitude of periodic component to be calculated.
        current_hour_in_day (float): Current hour in day for which periodic
                                     component is to be calculated.

    Returns:
        float: Periodic component for current hour in day.

    Raises:
        RuntimeError: If calculating the periodic component fails:
            * Invalid numeric types for scale, amplitude, or current hour
              (TypeError).
            * Invalid numeric values causing math computation errors
              (ValueError).
    """
    try:
        # Calculate the periodic component
        periodic_component = scale + amplitude * np.cos(
            2 * np.pi * (current_hour_in_day / TIME_HOURS_IN_DAY),
        )

        return periodic_component
    except (TypeError, ValueError) as e:
        msg = "Periodic component calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "scale": scale,
                "amplitude": amplitude,
                "current_hour_in_day": current_hour_in_day,
                "context": "Periodic component calculation",
            },
        )
        raise RuntimeError(msg) from e
