from typing import List, Tuple

import numpy as np

from pipeline.config.classes.Config import (
    Config,
)
from pipeline.data_generation.generation.requests.utils.request_generator_helper import (
    generate_requests_helper,
)
from pipeline.utils.logs.levels.info_logger import info


def generate_static_requests(
    config: Config,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate static requests and
    corresponding timestamps in hours.

    This function generates static requests and
    corresponding timestamps in hours.
    Static requests use a fixed Zipfian parameter, meaning
    the access distribution over keys does not change over time.

    Parameters:
        config (Config): Configuration object.

    Returns:
        Tuple[List[int], np.ndarray]: Keys requested and corresponding
                                      timestamps in hours.
    """
    # Use common helper to generate
    # requests based on a fixed alpha value
    requests, timestamps_hours = generate_requests_helper(config)

    info(
        f"{len(requests)} static requests and {len(timestamps_hours)} timestamps in hours generated"
    )

    return requests, timestamps_hours
