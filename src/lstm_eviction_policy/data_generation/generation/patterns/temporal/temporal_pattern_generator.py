import numpy as np

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.patterns.temporal.bursty_scale_setter import (
    set_bursty_scale,
)
from lstm_eviction_policy.data_generation.generation.patterns.temporal.periodic_component_calculator import (
    calculate_periodic_component,
)
from lstm_eviction_policy.data_generation.utils.seconds_to_hours_converter import (
    seconds_to_hours,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_temporal_pattern(current_seconds_in_day: float, config: Config) -> float:
    """
    Generate temporal pattern for current seconds in day.

    This function generates a delta time for the
    current seconds in day (i.e., gap between
    two consecutive requests), drawing a temporal pattern.
    This temporal pattern is determined as a combination of
    periodic and burst components, reflecting real-world data requests.

    Parameters:
        current_seconds_in_day (float): Current seconds in day for which
                                        temporal pattern (i.e., delta time)
                                        is to be generated.
        config (Config): Configuration object.

    Returns:
        float: Delta time for current seconds in day.
    """
    debug(
        f"Current seconds in day: {current_seconds_in_day} for which to calculate temporal pattern"
    )

    # Move from current seconds to
    # current hour in day
    current_hour_in_day = float(seconds_to_hours([current_seconds_in_day])[0])

    debug(
        f"Current hour in day: {current_hour_in_day} for which to calculate temporal pattern"
    )

    # Get scale and amplitude for
    # periodic component generation
    periodic_scale = config.pattern.temporal.periodic.scale
    periodic_amplitude = config.pattern.temporal.periodic.amplitude

    # Calculate a periodic component for
    # current hour in day
    periodic_component = calculate_periodic_component(
        periodic_scale, periodic_amplitude, current_hour_in_day
    )

    # Get burst high and low for burstiness,
    # as well as burst start and end hours
    burst_high = config.pattern.temporal.burstiness.high
    burst_low = config.pattern.temporal.burstiness.low
    burst_start_hour = config.pattern.temporal.burstiness.hours.start
    burst_end_hour = config.pattern.temporal.burstiness.hours.end

    # Set bursty scale
    bursty_scale = set_bursty_scale(
        burst_high, burst_low, burst_start_hour, burst_end_hour, current_hour_in_day
    )

    # Use the min between high and low
    # burst as minimum value for the
    # frequency scale to be calculated next
    min_freq_scale = min(burst_high, burst_low)

    debug(f"Minimum for frequency scale of temporal pattern: {min_freq_scale}")

    # Combine periodic component and bursty
    # scale to get the frequency scale (i.e., mean
    # interval in seconds between consecutive requests)
    freq_scale = max(min_freq_scale, periodic_component * bursty_scale)

    debug(f"Frequency scale for temporal pattern: {freq_scale}")

    # Draw the next inter-request time from an
    # exponential distribution with mean equals
    # frequency scale
    delta_t = np.random.exponential(scale=freq_scale)

    info(
        f"Delta time: {delta_t}, calculated for current hour in day: {current_hour_in_day}"
    )

    return delta_t
