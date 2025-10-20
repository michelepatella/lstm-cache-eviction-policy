from typing import List, Tuple

import numpy as np

from components.data.patterns.core.single_generator import (
    generate_single_pattern_request,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import (
    DATA_GENERATION_INITIAL_CURRENT_DAY,
    DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY,
    DATA_GENERATION_INITIAL_TIMESTAMP,
)
from pipeline.config.pydantic.config import Config


def generate_pattern_requests(
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    config: Config,
    time_step_duration: int = None,
    initial_timestamp: float = DATA_GENERATION_INITIAL_TIMESTAMP,
    initial_current_day: int = DATA_GENERATION_INITIAL_CURRENT_DAY,
    initial_current_seconds_in_day: int = DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY,
) -> Tuple[List[int], List[float]]:
    """
    Generate requests according to specific access and temporal patterns.

    This function generates requests along with their corresponding timestamps
    in seconds (i.e., absolute time of the requests), according to specific
    access and temporal patterns involving given keys, strongly affected by
    Zipfian distribution.

    Args:
        keys_range (np.ndarray): List of keys to generate requests for.
        zipf_probs (np.ndarray): List of Zipfian probabilities of the given keys.
        config (Config): Configuration object.
        time_step_duration (int): Time step to generate requests for.
        initial_timestamp (float): Initial timestamp in seconds.
        initial_current_day (int): Initial current day.
        initial_current_seconds_in_day (int): Initial seconds elapsed in
                                              the current day.

    Returns:
        Tuple[List[int], List[float]]:
            - requests: List of generated requests (key indices).
            - timestamps_seconds: Corresponding timestamps of the requests in seconds.

    Raises:
        RuntimeError: If generating pattern requests fails:
            * Invalid or empty keys range or Zipf probabilities
              (IndexError, ValueError, TypeError).
            * Invalid initial timestamp or current day/seconds values
              (TypeError, ValueError).
    """
    try:
        # Initialize data
        requests = []
        timestamps_seconds = [initial_timestamp]
        current_day = initial_current_day
        current_seconds_in_day = initial_current_seconds_in_day

        # Get the number of requests
        # to be generated
        num_requests = (
            time_step_duration
            if time_step_duration is not None
            else config.data.requests
        )
        debug(f"Number of requests to be generated: {num_requests}")

        # Define a seed to make the
        # generation process deterministic
        seed = config.data.seed
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
    except (IndexError, ValueError, TypeError) as e:
        msg = "Failed to generate pattern requests"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
