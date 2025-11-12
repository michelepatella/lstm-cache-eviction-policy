"""alpha_requests_generator.py

Module for generating requests for a single Zipfian alpha value.

This module provides the `generate_requests_for_alpha` function, which generates
a sequence of key requests and their corresponding timestamps in seconds,
based on a specific Zipfian alpha parameter. It calculates the Zipfian
probabilities for the given alpha and keys range, then generates requests
accordingly using the configured access and temporal patterns.

Functions:
    generate_requests_for_alpha(
        alpha: float,
        keys_range: ndarray,
        pipeline_config: PipelineConfig,
        time_step_duration: int | None = None
    ) -> tuple[list[int], list[float]]
        Generates a list of requested keys and their timestamps in seconds
        for a given Zipfian alpha value.
"""

import numpy as np

from components.const import LIST_FIRST_IDX, LIST_LAST_IDX
from components.data.patterns.core.generator import (
    generate_pattern_requests,
)
from components.logs.levels.debug_logger import debug
from components.math.zipf_probs_calculator import (
    calculate_zipf_probs,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def generate_requests_for_alpha(
    alpha: float,
    keys_range: np.ndarray,
    pipeline_config: PipelineConfig,
    time_step_duration: int | None = None,
) -> tuple[list[int], list[float]]:
    """Generate requests for a single alpha value.

    This function generates requests along with their
    timestamps for a single alpha value.

    Args:
        alpha (float): Alpha parameter for Zipfian distribution.
        keys_range (np.ndarray): List of available keys.
        pipeline_config (PipelineConfig): Configuration object.
        time_step_duration (int | None): Time step duration for generation.

    Returns:
        tuple[list[int], list[float]]:
            - requests: List of generated requests (key indices).
            - timestamps_seconds: Corresponding timestamps of the
                                  requests in seconds.
    """
    # Calculate Zipfian probabilities
    zipf_probs = calculate_zipf_probs(keys_range, alpha)

    debug(
        "Request generation for alpha started",
        extra={
            "alpha": alpha,
            "keys_range": [
                int(keys_range[LIST_FIRST_IDX]),
                int(keys_range[LIST_LAST_IDX]),
            ],
            "keys_num": len(keys_range),
            "time_step_duration": time_step_duration,
            "context": "Requests generation for alpha",
        },
    )

    # Generate requests and timestamps
    requests, timestamps_seconds = generate_pattern_requests(
        keys_range,
        zipf_probs,
        pipeline_config,
        time_step_duration=time_step_duration,
    )

    debug(
        "Request generation for alpha completed",
        extra={
            "alpha": alpha,
            "requests_generated_num": len(requests),
            "timestamps_seconds_generated_num": len(timestamps_seconds),
            "context": "Requests generation for alpha",
        },
    )

    return requests, timestamps_seconds
