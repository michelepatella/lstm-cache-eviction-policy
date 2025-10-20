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
    debug(f"Alpha to generate requests for: {alpha}")

    # Calculate Zipfian probabilities
    zipf_probs = calculate_zipf_probs(keys_range, alpha)

    # Generate requests and timestamps
    requests, timestamps_seconds = generate_pattern_requests(
        keys_range,
        zipf_probs,
        config,
        time_step_duration=time_step_duration,
    )

    debug(f"{len(requests)} requests generated for alpha: {alpha}")

    return requests, timestamps_seconds
