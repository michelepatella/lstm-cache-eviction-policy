from typing import List, Tuple

import numpy as np

from pipeline.config.classes.Config import Config
from components.data.requests.utils.alpha_requests_generator import (
    generate_requests_for_alpha,
)
from components.logs.levels.debug_logger import debug
from components.time.cyclic.seconds_to_hours_converter import (
    convert_seconds_to_hours_cyclic,
)


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

    Args:
        config (Config): Configuration object.
        alpha_range (List[float]): List of alpha parameters
                                   for dynamic requests.

    Returns:
        Tuple[List[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of the requests in hours.
    """
    # Retrieve keys range from configuration
    keys_config = config.data.generation.keys
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
        alpha_fixed = config.data.generation.pattern.access.zipf.alpha.fixed
        alpha_range = [alpha_fixed]
        time_step_duration = None
    else:
        # Otherwise, split requests
        # into several time steps
        num_requests = config.data.generation.requests
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
        # Generate requests for current alpha
        current_requests, current_timestamps_seconds = (
            generate_requests_for_alpha(
                alpha, keys_range, config, time_step_duration
            )
        )

        # Store generated requests and timestamps
        requests.extend(current_requests)
        timestamps_seconds.extend(current_timestamps_seconds)

    # Convert timestamps from seconds to hours
    timestamps_hours = convert_seconds_to_hours_cyclic(timestamps_seconds)

    return requests, timestamps_hours
