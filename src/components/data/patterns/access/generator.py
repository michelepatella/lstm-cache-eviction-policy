"""generator.py

Module for generating time-dependent access patterns.

This module provides the `generate_access_pattern` function, which selects
the next key to access based on the current time of day, past requests,
and multiple access patterns such as repetition, toggle, cycle, distortion,
and memory/fallback. The function dynamically adapts to the configured
behavior of each pattern and ensures continuity in simulated requests.

Functions:
    generate_access_pattern(
        zipf_probs: ndarray,
        keys_range: ndarray,
        current_abs_seconds: float,
        requests: list[int],
        config: Any
    ) -> int
        Determines the next key to access according to day-band-specific
        access patterns.
"""

from typing import Any

import numpy as np

from components.const import TIME_SECONDS_IN_DAY, TIME_SECONDS_IN_HOUR
from components.data.patterns.access.variants.cycle_generator import (
    generate_cycle_pattern,
)
from components.data.patterns.access.variants.distortion_generator import (
    generate_distortion_pattern,
)
from components.data.patterns.access.variants.memory_fallback_generator import (
    generate_memory_fallback_pattern,
)
from components.data.patterns.access.variants.repetition_generator import (
    generate_repetition_pattern,
)
from components.data.patterns.access.variants.toggle_generator import (
    generate_toggle_pattern,
)
from components.logs.levels.error_logger import error


def generate_access_pattern(
    zipf_probs: np.ndarray,
    keys_range: np.ndarray,
    current_abs_seconds: float,
    requests: list[int],
    config: Any,
) -> int:
    """Generate access pattern based on day band.

    This function determines which is the next key to be accessed according
    to multiple access patterns (repetition, toggle, cycle, distortion, memory,
    and fallback) based on day band.

    Args:
        zipf_probs (np.ndarray): List of Zipfian probabilities for keys.
        keys_range (np.ndarray): List of all available keys.
        current_abs_seconds (float): Current absolute time in seconds.
        requests (list[int]): List of keys requested so far.
        config (Any): Configuration object.

    Returns:
        int: Index of the next key to be accessed.

    Raises:
        RuntimeError: If generating the access pattern fails:
            * Accessing the keys range due to invalid or empty array
              (IndexError, ValueError).
            * Accessing configuration attributes due to missing or invalid
              fields (AttributeError).
            * Using invalid argument types in calculations or comparisons
              (TypeError).
    """
    try:
        # Get the current hour in day
        current_hour_in_day = (
            current_abs_seconds % TIME_SECONDS_IN_DAY
        ) / TIME_SECONDS_IN_HOUR

        # Prepare general configuration
        first_key = int(keys_range[0])
        num_keys = len(keys_range)
        requests_count = len(requests)

        behavior_config = config.data.patterns.access.behavior

        # Prepare repetition pattern configuration
        repetition_interval = behavior_config.repetition.interval
        repetition_offset = behavior_config.repetition.offset
        repetition_start = behavior_config.repetition.hours.start
        repetition_end = behavior_config.repetition.hours.end

        # Prepare toggle pattern configuration
        toggle_interval = behavior_config.toggle.interval
        toggle_first_base_request = behavior_config.toggle.base_requests.first
        toggle_second_base_request = (
            behavior_config.toggle.base_requests.second
        )
        toggle_forward = behavior_config.toggle.offsets.forward
        toggle_backward = behavior_config.toggle.offsets.backward
        toggle_start = behavior_config.toggle.hours.start
        toggle_end = behavior_config.toggle.hours.end

        # Prepare cycle pattern configuration
        cycle_base = behavior_config.cycle.base
        cycle_mod = behavior_config.cycle.mod
        cycle_divisor = behavior_config.cycle.divisor
        cycle_start = behavior_config.cycle.hours.start
        cycle_end = behavior_config.cycle.hours.end

        # Prepare distortion and noise patterns configuration
        distortion_interval = behavior_config.distortion.interval
        distortion_history = behavior_config.distortion.offsets.history
        distortion_correction = behavior_config.distortion.offsets.correction
        distortion_start = behavior_config.distortion.hours.start
        distortion_end = behavior_config.distortion.hours.end
        distortion_noise_min = behavior_config.distortion.noise.min
        distortion_noise_max = behavior_config.distortion.noise.max

        # Prepare memory pattern configuration
        memory_interval = behavior_config.memory.interval
        memory_offset = behavior_config.memory.offset

        # Generate pattern requests based on the
        # day band

        # Repetition pattern
        if (repetition_start <= current_hour_in_day < repetition_end) and (
            requests_count > repetition_offset
        ):
            requested_key = generate_repetition_pattern(
                repetition_interval,
                repetition_offset,
                requests_count,
                requests,
                keys_range,
                num_keys,
            )

        # Toggle pattern
        elif (toggle_start <= current_hour_in_day < toggle_end) and (
            requests_count
            > max(
                abs(toggle_backward),
                abs(toggle_forward),
            )
        ):
            requested_key = generate_toggle_pattern(
                toggle_interval,
                toggle_forward,
                toggle_backward,
                toggle_first_base_request,
                toggle_second_base_request,
                requests,
                requests_count,
                first_key,
                num_keys,
            )

        # Cycle pattern
        elif (cycle_start <= current_hour_in_day < cycle_end) and (
            requests_count > 0
        ):
            requested_key = generate_cycle_pattern(
                cycle_base,
                cycle_divisor,
                cycle_mod,
                requests_count,
                keys_range,
            )

        # Distortion pattern
        elif (distortion_start <= current_hour_in_day < distortion_end) and (
            requests_count > distortion_history
        ):
            requested_key = generate_distortion_pattern(
                distortion_interval,
                distortion_history,
                distortion_correction,
                requests,
                requests_count,
                num_keys,
                first_key,
                distortion_noise_min,
                distortion_noise_max,
            )

        # Memory/fallback pattern
        else:
            requested_key = generate_memory_fallback_pattern(
                memory_interval,
                memory_offset,
                requests,
                requests_count,
                keys_range,
                zipf_probs,
            )

        return requested_key
    except (IndexError, TypeError, AttributeError, ValueError) as e:
        msg = "Access pattern generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "zipf_probs_num": (
                    len(zipf_probs) if zipf_probs is not None else None
                ),
                "keys_range_len": (
                    len(keys_range) if keys_range is not None else None
                ),
                "current_abs_seconds": current_abs_seconds,
                "requests_count": (
                    len(requests) if requests is not None else None
                ),
                "first_request": requests[0] if requests else None,
                "context": "Access pattern generation",
            },
        )
        raise RuntimeError(msg) from e
