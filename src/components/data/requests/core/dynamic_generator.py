from typing import List, Tuple

import numpy as np

from components.data.requests.utils.generation_helper import (
    generate_requests_helper,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from pipeline.config.pydantic.config import Config


def generate_dynamic_requests(
    config: Config,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate dynamic requests and
    corresponding timestamps in hours.

    This function generates dynamic requests and
    corresponding timestamps in hours. Dynamic requests change
    over time: multiple alpha values are generated between min
    and max, and total requests are split into time steps.
    Each time step uses a different alpha, creating
    temporal variability in the access distribution.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[List[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of requests in hours.
    """
    # Retrieve Zipfian config
    zipf_config = config.data.generation.pattern.access.zipf
    alpha_min = zipf_config.alpha.min
    alpha_max = zipf_config.alpha.max
    steps = zipf_config.steps

    debug(
        f"Dynamic alpha values generation from"
        f" {alpha_min} to {alpha_max}, in {steps} steps"
    )

    # Generate evenly spaced alpha
    # values for dynamic time steps
    alpha_range = np.linspace(alpha_min, alpha_max, steps).tolist()

    debug(f"Alpha values generated for dynamic requests: {alpha_range}")

    # Use common helper to generate
    # requests based on dynamic alpha range
    requests, timestamps_hours = generate_requests_helper(
        config, alpha_range=alpha_range
    )

    info(
        f"{len(requests)} dynamic requests and"
        f" {len(timestamps_hours)} timestamps in hours generated"
    )

    return requests, timestamps_hours
