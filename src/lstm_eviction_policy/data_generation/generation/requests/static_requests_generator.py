import numpy as np

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.patterns.request_patterns_generator import (
    generate_pattern_requests,
)
from lstm_eviction_policy.data_generation.utils.timestamps_converter import (
    timestamps_seconds_to_hours,
)
from lstm_eviction_policy.data_generation.utils.zipf_props_calculator import (
    calculate_zipf_probs,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info


def generate_static_requests(config: Config) -> tuple[list[int], list[float]]:
    """
    Generate static requests and corresponding
    timestamps in hours.

    This function generates static requests to keys
    and corresponding timestamps in hours. The requests are generated
    according to specific patterns and are characterized by a
    Zipfian distribution to have more accesses to the first
    keys than to later ones. Zipfian probabilities of keys are fixed
    over time, resulting in static requests, thanks to the usage of
    a fixed Zipfian parameter.

    Parameters:
        config (Config): Configuration object.

    Returns:
        tuple[list[int], list[float]]: A tuple containing a list of
                                       integers (i.e., keys requested) and
                                       a list of floats (i.e., corresponding
                                       timestamps in hours).
    """
    # Retrieve the least and the greatest
    # keys (i.e., the first and the last ones)
    min_key = config.data.general.keys.min
    max_key = config.data.general.keys.max

    # Get the range of all possible keys
    keys_range = np.arange(min_key, max_key)

    debug(
        f"Static requests generation for keys range: [{min_key}, {max_key}] (tot: {len(keys_range)} keys)"
    )

    # Retrieve fixed Zipfian parameter
    fixed_alpha = config.data.pattern.access.zipf.alpha.fixed

    debug(f"Fixed Zipfian parameter for static requests generation: {fixed_alpha}")

    # Calculate Zipfian probabilities for all
    # the keys — ranging from min to max — given
    # a fixed Zipfian parameter
    zipf_probs = calculate_zipf_probs(keys_range, fixed_alpha)

    # Generate requests (i.e., couples of requested keys and
    # corresponding timestamps) according to specific patterns
    requests, timestamps_seconds = generate_pattern_requests(
        keys_range, zipf_probs, config
    )

    # Move from timestamps in seconds to
    # timestamps in hours
    timestamps_hours = timestamps_seconds_to_hours(timestamps_seconds)

    info(f"{len(requests)} static requests generated for {len(keys_range)} keys")

    return requests, timestamps_hours
