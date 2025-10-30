from typing import Any

import numpy as np

from components.data.requests.utils.generation_helper import (
    generate_requests_helper,
)
from components.logs.levels.info_logger import info


def generate_static_requests(
    config: Any,
) -> tuple[list[int], np.ndarray]:
    """Generate static requests and corresponding timestamps in hours.

    This function generates static requests and corresponding timestamps
    in hours. Static requests use a fixed Zipfian parameter, meaning
    the access distribution over keys does not change over time.

    Args:
        config (Any): Configuration object.

    Returns:
        tuple[list[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of requests in hours.
    """
    alpha_fixed = config.data.pattern.access.zipf.alpha.fixed

    info(
        "Static requests generation started",
        extra={
            "alpha_fixed": alpha_fixed,
            "context": "Static requests generation",
        },
    )

    # Use common helper to generate
    # requests based on a fixed alpha value
    requests, timestamps_hours = generate_requests_helper(
        [alpha_fixed],
        config,
    )

    info(
        "Static requests generation completed",
        extra={
            "requests_generated_num": len(requests),
            "timestamps_generated_num": len(timestamps_hours),
            "alpha_fixed": alpha_fixed,
            "context": "Static requests generation",
        },
    )

    return requests, timestamps_hours
