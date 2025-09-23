import numpy as np

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.patterns.access.patterns.cycle_pattern_generator import (
    generate_cycle_pattern,
)
from lstm_eviction_policy.data_generation.generation.patterns.access.patterns.distortion_pattern_generator import (
    generate_distortion_pattern,
)
from lstm_eviction_policy.data_generation.generation.patterns.access.patterns.memory_fallback_pattern_generator import (
    generate_memory_fallback_pattern,
)
from lstm_eviction_policy.data_generation.generation.patterns.access.patterns.repetition_pattern_generator import (
    generate_repetition_pattern,
)
from lstm_eviction_policy.data_generation.generation.patterns.access.patterns.toggle_pattern_generator import (
    generate_toggle_pattern,
)
from lstm_eviction_policy.data_generation.generation.utils.seconds_to_hours_converter import (
    seconds_to_hours,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_access_pattern(
    zipf_probs: np.ndarray,
    keys_range: np.ndarray,
    current_abs_seconds: float,
    requests: list[int],
    config: Config,
) -> int:
    """
    Generate access pattern based
    on day band.

    This function determines which is
    the next key to be accessed according
    to multiple access patterns (repetition,
    toggle, cycle, distortion, memory, and
    fallback) based on day band.

    Parameters:
        zipf_probs (np.ndarray): List of Zipfian probabilities
                                 for keys.
        keys_range (np.ndarray): List of all available keys.
        current_abs_seconds (float): Current absolute time in seconds.
        requests (list[int]): List of keys requested so far.
        config (Config): Configuration object.

    Returns:
        int: Index of the next key to be accessed.
    """
    # Get the current hour in day
    current_hour_in_day = float(seconds_to_hours([current_abs_seconds])[0])

    debug(f"Current hour in day for access pattern generation: {current_hour_in_day}")

    # Prepare general configuration
    first_key = int(keys_range[0])
    num_keys = len(keys_range)
    keys_range_size = len(keys_range) - 1
    requests_count = len(requests)

    debug(
        f"Requests generated so far: {requests_count}, for {num_keys} keys, ranging from {first_key} to {first_key+keys_range_size}"
    )

    behavior_config = config.data.pattern.access.behavior

    # Prepare repetition pattern configuration
    repetition_interval = behavior_config.repetition.interval
    repetition_offset = behavior_config.repetition.offset
    repetition_start = behavior_config.repetition.hours.start
    repetition_end = behavior_config.repetition.hours.end

    # Prepare toggle pattern configuration
    toggle_interval = behavior_config.toggle.interval
    toggle_first_base_request = behavior_config.toggle.base_requests.first
    toggle_second_base_request = behavior_config.toggle.base_requests.second
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

    # Prepare memory pattern configuration
    noise_min = behavior_config.noise.min
    noise_max = behavior_config.noise.max
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
        requests_count > max(abs(toggle_backward), abs(toggle_forward))
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
            keys_range_size,
        )

    # Cycle pattern
    elif (cycle_start <= current_hour_in_day < cycle_end) and (requests_count > 0):
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
            keys_range_size,
            first_key,
            noise_min,
            noise_max,
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

    info(f"Requests access pattern generated")

    return requested_key
