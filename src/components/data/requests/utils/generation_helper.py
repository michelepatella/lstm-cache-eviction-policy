from typing import List, Optional, Tuple

import numpy as np

from components.data.requests.utils.alpha_requests_generator import (
    generate_requests_for_alpha,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.cyclics.seconds_to_hours_converter import (
    convert_seconds_to_hours_cyclic,
)
from pipeline.config.pydantic.config import Config


def generate_requests_helper(
    alpha_range: List[float],
    config: Config,
    time_step_duration: Optional[int] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate requests according to static or dynamic Zipfian distributions.

    This helper function handles both static and dynamic request generation:
    - static: alpha range is None, uses fixed alpha
    - dynamic: alpha range is provided, splits total requests in time steps

    Args:
        alpha_range (Optional[List[float]]): List of alpha parameters.
        config (Config): Configuration object.
        time_step_duration (Optional[int]): Duration of each time step (None for
                                            static requests generation).

    Returns:
        Tuple[List[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of the requests in hours.

    Raises:
        RuntimeError: If generating requests fails:
            * Creating the keys range due to invalid min/max values
              (ValueError, TypeError).
            * Converting timestamps to hours due to invalid sequence or type
              (TypeError, ValueError).
    """
    try:
        # Retrieve keys range from configuration
        min_key = config.data.keys.min
        max_key = config.data.keys.max
        keys_range = np.arange(min_key, max_key + 1)

        debug(
            "Requests generation helper started",
            extra={
                "keys_range": [min_key, max_key],
                "keys_num": len(keys_range),
                "alpha_range": alpha_range,
                "time_step_duration": time_step_duration,
                "context": "Requests generation helper",
            },
        )

        # Iterate over alpha values
        # (static: one alpha, dynamic: multiple)
        requests = []
        timestamps_seconds = []
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

        debug(
            "Requests generation helper completed",
            extra={
                "requests_generated_num": len(requests),
                "timestamps_generated_num": len(timestamps_seconds),
                "context": "Requests generation helper",
            },
        )

        # Convert timestamps from seconds to hours
        timestamps_hours = convert_seconds_to_hours_cyclic(timestamps_seconds)

        return requests, timestamps_hours
    except (ValueError, TypeError) as e:
        msg = "Requests generation by helper failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "alpha_range": alpha_range,
                "time_step_duration": time_step_duration,
                "context": "Requests generation helper",
            },
        )
        raise RuntimeError(msg) from e
