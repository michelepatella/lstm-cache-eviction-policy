from typing import List, Tuple

import numpy as np

from pipeline.config.classes.Config import Config
from pipeline.const import SECONDS_IN_DAY
from components.data.patterns.access.generator import (
    generate_access_pattern,
)
from components.data.patterns.temporal.generator import (
    generate_temporal_pattern,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.time.cyclic.updater import (
    update_cyclic_time,
)


def generate_single_pattern_request(
    current_day: int,
    current_seconds_in_day: float,
    requests: List[int],
    keys_range: np.ndarray,
    zipf_probs: np.ndarray,
    config: Config,
) -> Tuple[int, float, float, int]:
    """
    Generate a single request and update the temporal state.

    This function generates one request based on the given access and
    temporal patterns, updates the current day and seconds in day, and
    computes the absolute timestamp of the request.

    Args:
        current_day (int): Current day in the simulation.
        current_seconds_in_day (float): Current seconds elapsed in the day.
        requests (List[int]): List of requests generated so far.
        keys_range (np.ndarray): Array of keys available for requests.
        zipf_probs (np.ndarray): Zipfian probabilities of keys.
        config (Config): Configuration object.

    Returns:
        Tuple[int, float, float, int]:
            - request: The generated key request.
            - absolute_seconds: Absolute timestamp of the request in seconds.
            - current_seconds_in_day: Updated seconds elapsed in the current day.
            - current_day: Updated day count in the simulation.
    """
    debug(
        f"Generating single request for day {current_day}, "
        f"seconds {current_seconds_in_day}"
    )

    # Generate delta time (gap between consecutive requests)
    delta_t = generate_temporal_pattern(current_seconds_in_day, config)

    # Update temporal state
    current_seconds_in_day, current_day = update_cyclic_time(
        current_seconds_in_day,
        current_day,
        SECONDS_IN_DAY,
        delta_t,
    )

    # Compute absolute timestamp
    absolute_seconds = current_day * SECONDS_IN_DAY + current_seconds_in_day

    # Generate access request
    request = generate_access_pattern(
        zipf_probs,
        keys_range,
        absolute_seconds,
        requests,
        config,
    )

    debug(
        f"Generated request {request} at "
        f"absolute seconds {absolute_seconds}"
    )

    info("Single request generated")

    return request, absolute_seconds, current_seconds_in_day, current_day
