from typing import List, Tuple

import numpy as np

from pipeline.config.classes.Config import Config
from pipeline.steps.data_generation.patterns.generator import (
    generate_pattern_requests,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from utils.math.zipf_probs_calculator import (
    calculate_zipf_probs,
)


def generate_requests_for_alpha(
    alpha: float,
    keys_range: np.ndarray,
    config: Config,
    time_step_duration: int = None,
) -> Tuple[List[int], List[float]]:
    """
    Generate requests for a single alpha value.

    This function generates requests along with their
    timestamps for a single alpha value.

    Args:
        alpha (float): Alpha parameter for Zipfian distribution.
        keys_range (np.ndarray): List of available keys.
        config (Config): Configuration object.
        time_step_duration (int): Time step duration for generation.

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

    info(f"{len(requests)} requests generated for alpha: {alpha}")

    return requests, timestamps_seconds
