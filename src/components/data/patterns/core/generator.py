import numpy as np

from components.const import (
    DATA_GENERATION_INITIAL_CURRENT_DAY,
    DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY,
    DATA_GENERATION_INITIAL_TIMESTAMP,
)
from components.data.patterns.core.single_generator import (
    generate_single_pattern_request,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config


def generate_pattern_requests(
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    config: Config,
    time_step_duration: int = None,
    initial_timestamp: float = DATA_GENERATION_INITIAL_TIMESTAMP,
    initial_current_day: int = DATA_GENERATION_INITIAL_CURRENT_DAY,
    initial_current_seconds_in_day: int = DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY,
) -> tuple[list[int], list[float]]:
    """Generate requests according to specific access and temporal patterns.

    This function generates requests along with their corresponding timestamps
    in seconds (i.e., absolute time of the requests), according to specific
    access and temporal patterns involving given keys, strongly affected by
    Zipfian distribution.

    Args:
        keys_range (np.ndarray): List of keys to generate requests for.
        zipf_probs (np.ndarray): List of Zipfian probabilities of the
                                 given keys.
        config (Config): Configuration object.
        time_step_duration (int): Time step to generate requests for.
        initial_timestamp (float): Initial timestamp in seconds.
        initial_current_day (int): Initial current day.
        initial_current_seconds_in_day (int): Initial seconds elapsed in
                                              the current day.

    Returns:
        tuple[list[int], list[float]]:
            - requests: List of generated requests (key indices).
            - timestamps_seconds: Corresponding timestamps of the
                                  requests in seconds.

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

        # Define a seed to make the
        # generation process deterministic
        seed = config.data.seed
        np.random.seed(seed)

        debug(
            "Pattern request generation started",
            extra={
                "requests_num": num_requests,
                "timestamp_initial": initial_timestamp,
                "current_day_initial": initial_current_day,
                "current_seconds_in_day_initial": initial_current_seconds_in_day,
                "keys_range_len": len(keys_range),
                "zipf_probs_sum": (
                    float(np.sum(zipf_probs))
                    if zipf_probs is not None
                    else None
                ),
                "seed": seed,
                "context": "Pattern request generation",
            },
        )

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

        debug(
            "Pattern request generation completed",
            extra={
                "requests_generated_num": len(requests),
                "timestamps_generated_num": len(timestamps_seconds),
                "context": "Pattern request generation",
            },
        )

        return requests, timestamps_seconds
    except (IndexError, ValueError, TypeError) as e:
        msg = "Pattern request generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "keys_range_len": (
                    len(keys_range) if keys_range is not None else 0
                ),
                "zipf_probs_sum": (
                    float(np.sum(zipf_probs))
                    if zipf_probs is not None
                    else None
                ),
                "requests_num": (
                    num_requests if "num_requests" in locals() else None
                ),
                "timestamp_initial": initial_timestamp,
                "current_day_initial": initial_current_day,
                "current_seconds_in_day_initial": initial_current_seconds_in_day,
                "context": "Pattern request generation",
            },
        )
        raise RuntimeError(msg) from e
