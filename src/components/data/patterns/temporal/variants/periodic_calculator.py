import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import HOURS_IN_DAY


def calculate_periodic_component(
    scale: int,
    amplitude: int,
    current_hour_in_day: float,
) -> float:
    """
    Calculate the periodic component given periodic scale and
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
            * Invalid numeric values causing math computation errors (ValueError).
    """
    try:
        debug(f"Scale for calculating periodic component: {scale}")
        debug(f"Amplitude for calculating periodic component: {amplitude}")

        # Calculate the periodic component
        periodic_component = scale + amplitude * np.cos(
            2 * np.pi * (current_hour_in_day / HOURS_IN_DAY)
        )

        debug(
            f"Periodic component: {periodic_component}, "
            f"calculated for hour in day: {current_hour_in_day}"
        )

        return periodic_component
    except (TypeError, ValueError) as e:
        msg = "Failed to calculate periodic component"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
