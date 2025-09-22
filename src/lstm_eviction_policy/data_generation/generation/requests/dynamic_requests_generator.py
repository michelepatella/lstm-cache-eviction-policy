import numpy as np

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.patterns.request_patterns_generator import (
    generate_pattern_requests,
)
from lstm_eviction_policy.data_generation.generation.utils.seconds_to_hours_converter import (
    seconds_to_hours,
)
from lstm_eviction_policy.data_generation.generation.utils.zipf_props_calculator import (
    calculate_zipf_probs,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_dynamic_requests(config: Config) -> tuple[list[int], np.ndarray]:
    # Retrieve general data configuration
    num_requests = config.data.general.requests.count

    debug(f"Total number of requests to be generated: {num_requests}")

    keys_config = config.data.general.keys

    # Retrieve the least and the greatest
    # keys (i.e., the first and the last ones)
    min_key = keys_config.min
    max_key = keys_config.max

    # Get the range of all possible keys
    keys_range = np.arange(min_key, max_key)

    debug(
        f"Dynamic requests generation for keys range: [{min_key}, {max_key}] (tot: {len(keys_range)} keys)"
    )

    zipf_config = config.data.pattern.access.zipf
    # Retrieve Zipfian configurations
    alpha_min = zipf_config.alpha.min
    alpha_max = zipf_config.alpha.max
    steps = zipf_config.steps

    debug(f"Boundaries for alpha values generation: {alpha_min}, {alpha_max}")
    debug(f"Total steps for alpha values generation: {steps}")

    # Generate a total number of alpha parameters (i.e.,
    # Zipfian distribution parameters) — ranging
    # from alpha min to max — equal to specified steps
    alpha_range = np.linspace(
        alpha_min,
        alpha_max,
        steps,
    )

    debug(f"Alpha values generated: {alpha_range}")

    # Calculate how many times
    # each time step will last
    # (within each time step a different
    # alpha value will be used)
    time_step_duration = num_requests // steps

    debug(f"Time step duration for dynamic data generation: {time_step_duration}")

    # Initialize data
    requests = []
    timestamps_seconds = []

    # For each alpha value generated
    for current_alpha in alpha_range:
        # Calculate Zipfian probabilities for all
        # the keys — ranging from min to max — given
        # the current Zipfian parameter
        zipf_probs = calculate_zipf_probs(keys_range, current_alpha)

        # Generate requests (i.e., couples of requested keys and
        # corresponding timestamps) according to specific patterns,
        # within the predefined, current time step duration
        current_requests, current_timestamps_seconds = generate_pattern_requests(
            keys_range, zipf_probs, config, time_step_duration=time_step_duration
        )

        info(
            f"{len(current_requests)} requests generated for dynamic alpha value: {current_alpha}"
        )

        # Store requests and corresponding
        # timestamps generated for the
        # current time step
        requests.extend(current_requests)
        timestamps_seconds.extend(current_timestamps_seconds)

    # Move from timestamps in seconds to
    # timestamps in hours
    timestamps_hours = seconds_to_hours(timestamps_seconds)

    info(f"{len(requests)} dynamic requests generated for {len(keys_range)} keys")

    return requests, timestamps_hours
