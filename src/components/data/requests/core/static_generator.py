from typing import List, Tuple

import numpy as np

from components.data.requests.utils.generation_helper import (
    generate_requests_helper,
)
from components.logs.levels.info_logger import info
from pipeline.config.pydantic.config import Config


def generate_static_requests(
    config: Config,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate static requests and corresponding timestamps in hours.

    This function generates static requests and corresponding timestamps
    in hours. Static requests use a fixed Zipfian parameter, meaning
    the access distribution over keys does not change over time.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[List[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of requests in hours.
    """
    alpha_fixed = config.data.pattern.access.zipf.alpha.fixed

    # Use common helper to generate
    # requests based on a fixed alpha value
    requests, timestamps_hours = generate_requests_helper([alpha_fixed], config)

    info(
        f"{len(requests)} static requests and "
        f"{len(timestamps_hours)} timestamps in hours generated"
    )

    return requests, timestamps_hours
