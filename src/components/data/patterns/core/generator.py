"""generator.py

Module for generating requests and timestamps according to access patterns.

This module provides the `generate_pattern_requests` function, which
produces a sequence of key requests along with their corresponding timestamps
in absolute seconds. The requests are generated based on specified access
patterns, day/time behavior, and Zipfian distribution.

Functions:
    generate_pattern_requests(
        keys_range: ndarray,
        zipf_probs: ndarray,
        pipeline_config: PipelineConfig,
        time_step_duration: int | None = None,
        timestamps_start: float = DATA_GENERATION_TIMESTAMPS_START,
        current_day_start: int = DATA_GENERATION_CURRENT_DAY_START,
        current_seconds_in_day_start: int = DATA_GENERATION_CURRENT_SECONDS_IN_DAY_START
    ) -> tuple[list[int], list[float]]
        Generates requests and timestamps according to the configured access
        patterns and Zipfian probabilities.
"""

import numpy as np

from components.const import (
    DATA_GENERATION_CURRENT_DAY_START,
    DATA_GENERATION_CURRENT_SECONDS_IN_DAY_START,
    DATA_GENERATION_TIMESTAMPS_START,
)
from components.data.patterns.core.single_generator import (
    generate_single_pattern_request,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def generate_pattern_requests(
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    pipeline_config: PipelineConfig,
    time_step_duration: int = None,
    timestamps_start: float = DATA_GENERATION_TIMESTAMPS_START,
    current_day_start: int = DATA_GENERATION_CURRENT_DAY_START,
    current_seconds_in_day_start: int = DATA_GENERATION_CURRENT_SECONDS_IN_DAY_START,
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
        pipeline_config (PipelineConfig): Configuration object.
        time_step_duration (int): Time step to generate requests for.
        timestamps_start (float): Initial timestamp in seconds.
        current_day_start (int): Initial current day.
        current_seconds_in_day_start (int): Initial seconds elapsed in
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
        timestamps_seconds = [timestamps_start]
        current_day = current_day_start
        current_seconds_in_day = current_seconds_in_day_start

        # Get the number of requests
        # to be generated
        num_requests = (
            time_step_duration
            if time_step_duration is not None
            else pipeline_config.data.general.requests
        )

        debug(
            "Pattern request generation started",
            extra={
                "requests_num": num_requests,
                "timestamps_start": timestamps_start,
                "current_day_start": current_day_start,
                "current_seconds_in_day_start": current_seconds_in_day_start,
                "keys_range_len": len(keys_range),
                "zipf_probs_sum": (
                    float(np.sum(zipf_probs))
                    if zipf_probs is not None
                    else None
                ),
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
                    pipeline_config,
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
                "timestamps_seconds_generated_num": len(timestamps_seconds),
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
                "timestamps_start": timestamps_start,
                "current_day_start": current_day_start,
                "current_seconds_in_day_start": current_seconds_in_day_start,
                "context": "Pattern request generation",
            },
        )
        raise RuntimeError(msg) from e
