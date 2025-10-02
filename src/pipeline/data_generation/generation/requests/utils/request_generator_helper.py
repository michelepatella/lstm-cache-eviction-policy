from typing import List, Tuple

import numpy as np

from const import PERIOD, SECONDS_IN_HOUR
from pipeline.config.classes.Config import Config
from pipeline.data_generation.generation.patterns.request_patterns_generator import (
    generate_pattern_requests,
)
from pipeline.data_generation.generation.utils.zipf_props_calculator import (
    calculate_zipf_probs,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info


def generate_requests_helper(
    config: Config,
    alpha_range: List[float] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate requests according
    to static or dynamic Zipfian distributions.

    This helper function handles both static
    and dynamic request generation:
    - static: alpha range is None, uses fixed alpha
    - dynamic: alpha range is provided, splits total
               requests in time steps

    Parameters:
        config (Config): Configuration object.
        alpha_range (List[float]): Optional list of alpha parameters
                                   for dynamic requests.

    Returns:
        Tuple[List[int], np.ndarray]: Generated requests and
                                      timestamps in hours.
    """
    # Retrieve keys range from configuration
    keys_config = config.data.general.keys
    min_key = keys_config.min
    max_key = keys_config.max
    keys_range = np.arange(min_key, max_key + 1)

    debug(
        f"Requests generation for keys range: [{min_key},"
        f" {max_key}] (total: {len(keys_range)} keys)"
    )

    # If no alpha range is provided
    if alpha_range is None:
        # Use static fixed alpha and
        # don't consider any time step duration
        alpha_fixed = config.data.pattern.access.zipf.alpha.fixed
        alpha_range = [alpha_fixed]
        time_step_duration = None
    else:
        # Otherwise, split requests
        # into several time steps
        num_requests = config.data.general.requests.count
        time_step_duration = num_requests // len(alpha_range)

        debug(
            f"Time step duration for dynamic "
            f"data generation: {time_step_duration}"
        )

    requests = []
    timestamps_seconds = []

    # Iterate over alpha values
    # (static: one alpha, dynamic: multiple)
    for alpha in alpha_range:
        # Calculate Zipfian probabilities
        # for keys given current alpha
        zipf_probs = calculate_zipf_probs(keys_range, alpha)

        # Generate requests and timestamps
        # for current alpha / time step
        (
            current_requests,
            current_timestamps_seconds,
        ) = generate_pattern_requests(
            keys_range,
            zipf_probs,
            config,
            time_step_duration=time_step_duration,
        )

        info(
            f"{len(current_requests)} requests generated "
            f"for alpha value: {alpha}"
        )

        # Store generated requests and timestamps
        requests.extend(current_requests)
        timestamps_seconds.extend(current_timestamps_seconds)

    # Convert timestamps from seconds to hours
    timestamps_hours = (
        np.array(timestamps_seconds) % PERIOD
    ) / SECONDS_IN_HOUR

    return requests, timestamps_hours
