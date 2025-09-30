from typing import List, Tuple

import numpy as np

from const import PERIOD
from pipeline.config.classes.Config import (
    Config,
)
from pipeline.data_generation.generation.patterns.access.access_pattern_generator import (
    generate_access_pattern,
)
from pipeline.data_generation.generation.patterns.temporal.temporal_pattern_generator import (
    generate_temporal_pattern,
)
from pipeline.data_generation.generation.patterns.utils.time_updater import (
    update_time,
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

    Parameters:
        keys_range (np.ndarray): List of keys to generate requests for.
        zipf_probs (np.ndarray): List of Zipfian probabilities of the given keys.
        config (Config): Configuration object.
        time_step_duration (int): Optional time step to generate requests for.

    Returns:
        Tuple[List[int], List[float]]: List of requests along with their
                                       corresponding timestamps.
    """
    # Initialize data
    requests = []
    timestamps_seconds = [0.0]  # Get start from timestamp zero
    current_day = 0  # Get start from day zero
    current_seconds_in_day = 0.0  # Get start from midnight (second zero)

    general_data_config = config.data.general

    # Get the number of requests
    # to be generated
    num_requests = (
        time_step_duration
        if time_step_duration is not None
        else general_data_config.requests.count
    )

    debug(f"Number of requests to be generated: {num_requests}")

    debug(f"Period of requests generation: {PERIOD}")

    # Define a seed to make the
    # generation process deterministic
    seed = general_data_config.seed
    np.random.seed(seed)

    debug(f"Seed for requests generation: {seed}")

    # For each request to be generated
    for _ in range(num_requests):
        debug(
            f"Request generation for day {current_day}, seconds: {current_seconds_in_day}"
        )

        # Generate delta time (i.e., gap between
        # two consecutive requests in seconds)
        delta_t = generate_temporal_pattern(current_seconds_in_day, config)

        # Update time (current seconds in day and
        # current day)
        current_seconds_in_day, current_day = update_time(
            current_seconds_in_day,
            current_day,
            PERIOD,
            delta_t,
        )

        # Get current absolute seconds of the
        # request to be generated
        current_abs_seconds = current_day * PERIOD + current_seconds_in_day

        # Generate a request (i.e., accessed key)
        request = generate_access_pattern(
            zipf_probs,
            keys_range,
            current_abs_seconds,
            requests,
            config,
        )

        # Store new request and corresponding
        # timestamp in seconds (absolute seconds)
        requests.append(request)
        timestamps_seconds.append(current_abs_seconds)

        debug(
            f"Generated request {request} at absolute seconds {current_abs_seconds}"
        )

    info("Pattern requests generated")

    return requests, timestamps_seconds
