import numpy as np

from pipeline.const import HOURS_IN_DAY
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info


def calculate_periodic_component(
    scale: int,
    amplitude: int,
    current_hour_in_day: float,
) -> float:
    """
    Calculate the periodic component given
    periodic scale and amplitude, as well as current hour in day.

    This function calculates a periodic component for
    current hour in day, given scale and amplitude, by adding
    the base scale to the cosine-modulated amplitude.

    Args:
        scale (int): Scale of periodic component to be calculated.
        amplitude (int): Amplitude of periodic component to be calculated.
        current_hour_in_day (float): Current hour in day for which periodic
                                     component is to be calculated.

    Returns:
        float: Periodic component for current hour in day.
    """
    debug(f"Scale for calculating periodic component: {scale}")
    debug(f"Amplitude for calculating periodic component: {amplitude}")

    # Calculate the periodic component
    periodic_component = scale + amplitude * np.cos(
        2 * np.pi * (current_hour_in_day / HOURS_IN_DAY)
    )

    info(
        f"Periodic component: {periodic_component}, "
        f"calculated for hour in day: {current_hour_in_day}"
    )

    return periodic_component
