import numpy as np

from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    info,
)


def calculate_zipf_probs(keys: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate Zipfian probabilities for the given keys,
    given a specific Zipfian parameter.

    This function calculates the Zipfian probabilities
    for all the given keys, using alpha parameter as Zipfian parameter.
    The resulting probabilities — which will sum to 1 — of the first keys
    will be greater than those of later ones, according to the Zipfian
    distribution governed by alpha parameter.

    Parameters:
        keys (np.ndarray): Keys to calculate Zipfian probabilities for.
        alpha (float): The Zipfian parameter which governs its distribution.

    Returns:
        np.ndarray: List of Zipfian probabilities — which sum to 1 — for all
              the given keys.
    """
    debug(f"Zipfian parameter for probabilities calculation: {alpha}")

    # Calculate the probability of the keys
    # according to the Zipf's distribution formula
    zipf_probs = 1.0 / np.power(keys, alpha)

    debug(
        f"(Before normalization) Zipfian probabilities range: [{np.min(zipf_probs)}, {np.max(zipf_probs)}]"
    )

    # Normalize probabilities to
    # make sum to 1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)

    debug(
        f"(After normalization) Zipfian probabilities range: [{np.min(zipf_probs_normalized)}, {np.max(zipf_probs_normalized)}]"
    )
    debug(
        f"Sum of Zipfian key probabilities after normalization: {np.sum(zipf_probs_normalized)}"
    )

    info(
        f"{len(zipf_probs_normalized)} Zipfian probabilities calculated for {len(keys)} keys"
    )

    return zipf_probs_normalized
