from typing import List, Optional, Tuple

import numpy as np

from components.data.patterns.core.generator import (
    generate_pattern_requests,
)
from components.logs.levels.debug_logger import debug
from components.math.zipf_probs_calculator import (
    calculate_zipf_probs,
)
from pipeline.config.pydantic.config import Config


def generate_requests_for_alpha(
    alpha: float,
    keys_range: np.ndarray,
    config: Config,
    time_step_duration: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """
    Generate requests for a single alpha value.

    This function generates requests along with their
    timestamps for a single alpha value.

    Args:
        alpha (float): Alpha parameter for Zipfian distribution.
        keys_range (np.ndarray): List of available keys.
        config (Config): Configuration object.
        time_step_duration (Optional[int]): Time step duration for generation.

    Returns:
        Tuple[List[int], List[float]]:
            - requests: List of generated requests (key indices).
            - timestamps_seconds: Corresponding timestamps of the requests in seconds.
    """
    # Calculate Zipfian probabilities
    zipf_probs = calculate_zipf_probs(keys_range, alpha)

    debug(
        "Request generation for alpha started",
        extra={
            "alpha": alpha,
            "keys_range": [int(keys_range[0]), int(keys_range[-1])],
            "num_keys": len(keys_range),
            "time_step_duration": time_step_duration,
            "context": "Requests generation for alpha",
        },
    )

    # Generate requests and timestamps
    requests, timestamps_seconds = generate_pattern_requests(
        keys_range,
        zipf_probs,
        config,
        time_step_duration=time_step_duration,
    )

    debug(
        "Request generation for alpha completed",
        extra={
            "alpha": alpha,
            "num_requests_generated": len(requests),
            "num_timestamps_generated": len(timestamps_seconds),
            "context": "Requests generation for alpha",
        },
    )

    return requests, timestamps_seconds
