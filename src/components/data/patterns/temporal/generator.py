"""generator.py

Module for generating temporal patterns for synthetic requests.

This module provides the `generate_temporal_pattern` function, which
produces a delta time (gap between consecutive requests) based on
configured periodic and burstiness patterns, simulating real-world
request temporal dynamics.

Functions:
    generate_temporal_pattern(
        current_seconds_in_day: float,
        pipeline_config: PipelineConfig
    ) -> float
        Generates the next inter-request time (delta_t) based on current
        time of day and configuration for periodic and burstiness patterns.
"""

from typing import Any

import numpy as np

from components.const import TIME_SECONDS_IN_HOUR
from components.data.patterns.temporal.variants.burst_setter import (
    set_bursty_scale,
)
from components.data.patterns.temporal.variants.periodic_calculator import (
    calculate_periodic_component,
)
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def generate_temporal_pattern(
    current_seconds_in_day: float,
    pipeline_config: PipelineConfig,
) -> float:
    """Generate temporal pattern for current seconds in day.

    This function generates a delta time for the current seconds
    in day (i.e., gap between two consecutive requests), drawing a
    temporal pattern. This temporal pattern is determined as a
    combination of periodic and burst components, reflecting real-world
    data requests.

    Args:
        current_seconds_in_day (float): Current seconds in day for which
                                        temporal pattern (i.e., delta time)
                                        is to be generated.
        pipeline_config (PipelineConfig): Configuration object.

    Returns:
        float: Delta time for current seconds in day.

    Raises:
        RuntimeError: If generating the temporal pattern fails:
            * Invalid current seconds in day or calculation of current hour
              (TypeError, ValueError).
            * Invalid configuration values for periodic or burst components
              (AttributeError, TypeError, ValueError).
            * Random number generation for delta time fails
              (ValueError, TypeError).
    """
    try:
        # Move from current seconds to
        # current hour in day
        current_hour_in_day = current_seconds_in_day / TIME_SECONDS_IN_HOUR

        periodic_pattern_config = (
            pipeline_config.data.synthetic.patterns.temporal.periodic
        )
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

        burstiness_pattern_config = (
            pipeline_config.data.synthetic.patterns.temporal.burstiness
        )
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

        # Combine periodic component and bursty
        # scale to get the frequency scale (i.e., mean
        # interval in seconds between consecutive requests)
        freq_scale = max(
            min_freq_scale,
            periodic_component * bursty_scale,
        )

        # Draw the next inter-request time from an
        # exponential distribution with mean equals
        # frequency scale
        delta_t = np.random.exponential(scale=freq_scale)

        return delta_t
    except (ValueError, TypeError, AttributeError) as e:
        msg = "Temporal pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "current_seconds_in_day": current_seconds_in_day,
                "current_hour_in_day": (
                    current_seconds_in_day / TIME_SECONDS_IN_HOUR
                    if isinstance(current_seconds_in_day, (int, float))
                    else None
                ),
                "context": "Temporal pattern generation",
            },
        )
        raise RuntimeError(msg) from e
