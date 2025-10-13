from typing import List, Tuple

import numpy as np

from const import (
    DATA_GENERATION_INITIAL_CURRENT_DAY,
    DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY,
    DATA_GENERATION_INITIAL_TIMESTAMP,
)
from pipeline.config.classes.Config import Config
from pipeline.steps.data_generation.patterns.utils.single_pattern_request_generator import (
    generate_single_pattern_request,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info


def generate_pattern_requests(
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    config: Config,
    time_step_duration: int = None,
) -> Tuple[List[int], List[float]]:
    """
    Generate requests according to
    specific access and temporal patterns.

    This function generates requests along
    with their corresponding timestamps in seconds
    (i.e., absolute time of the requests), according
    to specific access and temporal patterns involving
    given keys, strongly affected by Zipfian distribution.

    Args:
        keys_range (np.ndarray): List of keys to generate requests for.
        zipf_probs (np.ndarray): List of Zipfian probabilities
                                 of the given keys.
        config (Config): Configuration object.
        time_step_duration (int): Optional time step to generate requests for.

    Returns:
        Tuple[List[int], List[float]]: List of requests along with their
                                       corresponding timestamps.
    """
    # Initialize data
    requests = []
    timestamps_seconds = DATA_GENERATION_INITIAL_TIMESTAMP
    current_day = DATA_GENERATION_INITIAL_CURRENT_DAY
    current_seconds_in_day = DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY

    general_data_config = config.data.generation

    # Get the number of requests
    # to be generated
    num_requests = (
        time_step_duration
        if time_step_duration is not None
        else general_data_config.requests
    )

    debug(f"Number of requests to be generated: {num_requests}")

    # Define a seed to make the
    # generation process deterministic
    seed = general_data_config.seed
    np.random.seed(seed)

    debug(f"Seed for requests generation: {seed}")

    # For each request to be generated
    for _ in range(num_requests):
        # Generate the single request
        request, absolute_seconds, current_seconds_in_day, current_day = (
            generate_single_pattern_request(
                current_day,
                current_seconds_in_day,
                requests,
                keys_range,
                zipf_probs,
                config,
            )
        )

        # Store new request and corresponding
        # timestamp in seconds (absolute seconds)
        requests.append(request)
        timestamps_seconds.append(absolute_seconds)

    info("Pattern requests generated")

    return requests, timestamps_seconds
