import numpy as np

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def calculate_zipf_probs(items: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate Zipfian probabilities for the given items,
    given a specific Zipfian parameter.

    This function calculates the Zipfian probabilities
    for all the given items, using alpha parameter as Zipfian parameter.
    The resulting probabilities — which will sum to 1 — of the first items
    will be greater than those of later ones, according to the Zipfian
    distribution governed by alpha parameter.

    Args:
        items (np.ndarray): Items to calculate Zipfian probabilities for.
        alpha (float): The Zipfian parameter which governs its distribution.

    Returns:
        np.ndarray: List of Zipfian probabilities — which sum to 1 —
                    for all the given items.
    """
    debug(f"Zipfian parameter for probabilities calculation: {alpha}")

    # Calculate the probability of the items
    # according to the Zipf's distribution formula
    zipf_probs = 1.0 / np.power(items, alpha)

    debug(
        f"(Before normalization) Zipfian probabilities "
        f"range: [{np.min(zipf_probs)}, {np.max(zipf_probs)}]"
    )

    # Normalize probabilities to
    # make sum to 1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)

    debug(
        f"(After normalization) Zipfian probabilities"
        f" range: [{np.min(zipf_probs_normalized)},"
        f" {np.max(zipf_probs_normalized)}]"
    )
    debug(
        f"Sum of Zipfian items probabilities"
        f" after normalization: {np.sum(zipf_probs_normalized)}"
    )

    info(
        f"{len(zipf_probs_normalized)} Zipfian "
        f"probabilities calculated for {len(items)} items"
    )

    return zipf_probs_normalized
