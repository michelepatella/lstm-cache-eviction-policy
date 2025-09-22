import numpy as np

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.requests.utils.request_generator_helper import (
    generate_requests_helper,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_dynamic_requests(config: Config) -> tuple[list[int], np.ndarray]:
    """
    Generate dynamic requests and
    corresponding timestamps in hours.

    This function generates dynamic requests and
    corresponding timestamps in hours. Dynamic requests change
    over time: multiple alpha values are generated between min
    and max, and total requests are split into time steps.
    Each time step uses a different alpha, creating
    temporal variability in the access distribution.

    Parameters:
        config (Config): Configuration object.

    Returns:
        tuple[list[int], np.ndarray]: Keys requested and corresponding
                                      timestamps in hours.
    """
    # Retrieve Zipfian config
    zipf_config = config.data.pattern.access.zipf
    alpha_min = zipf_config.alpha.min
    alpha_max = zipf_config.alpha.max
    steps = zipf_config.steps

    debug(
        f"Dynamic alpha values generation from {alpha_min} to {alpha_max}, in {steps} steps."
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
        f"{len(requests)} dynamic requests and {len(timestamps_hours)} timestamps in hours generated"
    )

    return requests, timestamps_hours
