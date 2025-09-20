import numpy as np
from data_generation.generation.patterns.request_patterns_generator import (
    generate_pattern_requests,
)
from data_generation.utils.timestamp_converter import timestamps_to_hours
from data_generation.utils.zipf_calculator import calculate_zipf_probs
from utils.logs.log_utils import debug, info


def generate_static_requests(config_settings):
    """
    Method to orchestrate the static requests' generation.
    :param config_settings: The configuration settings.
    :return: Static requests and timestamps generated.
    """
    # initial message
    info("🔄 Static requests generation started...")

    # calculate the probabilities
    probs = calculate_zipf_probs(
        np.arange(config_settings.first_key, config_settings.last_key),
        config_settings.zipf_alpha,
    )

    timestamps = [0]
    # generate patterns
    requests, timestamps = generate_pattern_requests(
        probs, config_settings.num_requests, timestamps, config_settings
    )

    timestamps = timestamps_to_hours(timestamps)

    # debugging
    debug(f"⚙️Probabilities length: {len(probs)}.")
    debug(f"⚙️Requests length: {len(requests)}.")
    debug(f"⚙️Timestamps length: {len(timestamps)}.")

    # show a successful message
    info("🟢 Static requests generated.")

    return requests, timestamps
