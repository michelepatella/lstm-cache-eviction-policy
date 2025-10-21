import numpy as np

from components.const import TIME_SECONDS_IN_HOUR
from components.data.patterns.temporal.variants.burst_setter import (
    set_bursty_scale,
)
from components.data.patterns.temporal.variants.periodic_calculator import (
    calculate_periodic_component,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config


def generate_temporal_pattern(
    current_seconds_in_day: float, config: Config
) -> float:
    """
    Generate temporal pattern for current seconds in day.

    This function generates a delta time for the current seconds
    in day (i.e., gap between two consecutive requests), drawing a
    temporal pattern. This temporal pattern is determined as a
    combination of periodic and burst components, reflecting real-world
    data requests.

    Args:
        current_seconds_in_day (float): Current seconds in day for which
                                        temporal pattern (i.e., delta time)
                                        is to be generated.
        config (Config): Configuration object.

    Returns:
        float: Delta time for current seconds in day.

    Raises:
        RuntimeError: If generating the temporal pattern fails:
            * Invalid current seconds in day or calculation of current hour
              (TypeError, ValueError).
            * Invalid configuration values for periodic or burst components
              (AttributeError, TypeError, ValueError).
            * Random number generation for delta time fails (ValueError, TypeError).
    """
    try:
        debug(
            f"Current seconds in day: {current_seconds_in_day}"
            f" for temporal pattern generation"
        )

        # Move from current seconds to
        # current hour in day
        current_hour_in_day = current_seconds_in_day / TIME_SECONDS_IN_HOUR
        debug(
            f"Current hour in day: {current_hour_in_day} "
            f"for temporal pattern generation"
        )

        periodic_pattern_config = config.data.pattern.temporal.periodic
        # Get scale and amplitude for
        # periodic component generation
        periodic_scale = periodic_pattern_config.scale
        periodic_amplitude = periodic_pattern_config.amplitude

        # Calculate a periodic component for
        # current hour in day
        periodic_component = calculate_periodic_component(
            periodic_scale,
            periodic_amplitude,
            current_hour_in_day,
        )

        burstiness_pattern_config = config.data.pattern.temporal.burstiness
        # Get burst high and low for burstiness,
        # as well as burst start and end hours
        burst_high = burstiness_pattern_config.high
        burst_low = burstiness_pattern_config.low
        burst_start_hour = burstiness_pattern_config.hours.start
        burst_end_hour = burstiness_pattern_config.hours.end

        # Set bursty scale
        bursty_scale = set_bursty_scale(
            burst_high,
            burst_low,
            burst_start_hour,
            burst_end_hour,
            current_hour_in_day,
        )

        # Use the min between high and low
        # burst as minimum value for the
        # frequency scale to be calculated next
        min_freq_scale = min(burst_high, burst_low)
        debug(
            f"Minimum for frequency scale of temporal pattern: {min_freq_scale}"
        )

        # Combine periodic component and bursty
        # scale to get the frequency scale (i.e., mean
        # interval in seconds between consecutive requests)
        freq_scale = max(
            min_freq_scale,
            periodic_component * bursty_scale,
        )

        debug(f"Frequency scale for temporal pattern: {freq_scale}")

        # Draw the next inter-request time from an
        # exponential distribution with mean equals
        # frequency scale
        delta_t = np.random.exponential(scale=freq_scale)

        debug(
            f"Delta time: {delta_t}, calculated for"
            f" current hour in day: {current_hour_in_day}"
        )

        return delta_t
    except (ValueError, TypeError, AttributeError) as e:
        msg = "Failed to generate temporal pattern"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
