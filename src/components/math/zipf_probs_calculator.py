import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_zipf_probs(items: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate Zipf probabilities for the items.

    This function calculates the Zipf probabilities for all the
    given items, using alpha as Zipf parameter. The resulting
    probabilities of the first items will be greater than those of
    later ones.

    Args:
        items (np.ndarray): Items to calculate Zipf probabilities for.
        alpha (float): The Zipf parameter.

    Returns:
        np.ndarray: List of Zipf probabilities for all the given items.

    Raises:
        RuntimeError: If calculation of Zipf probabilities fails:
            * If items is not a valid NumPy array or contains invalid
              types (TypeError).
            * If items contains zeros leading to division by zero
              (ZeroDivisionError).
            * If normalization fails due to empty array or invalid operations
              (ValueError).
    """
    try:
        debug(
            f"Number of items for Zipf probabilities calculation: {len(items)}"
        )
        debug(f"Zipf parameter for probabilities calculation: {alpha}")

        # Calculate the probability of the items
        # according to the Zipf's distribution formula
        zipf_probs = 1.0 / np.power(items, alpha)
        debug(
            f"(Before normalization) Zipf probabilities "
            f"range: [{np.min(zipf_probs)}, {np.max(zipf_probs)}]"
        )

        # Normalize probabilities to make sum to 1
        zipf_probs_norm = zipf_probs / np.sum(zipf_probs)
        debug(
            f"(After normalization) Zipf probabilities"
            f" range: [{np.min(zipf_probs_norm)},"
            f" {np.max(zipf_probs_norm)}]"
        )
        debug(
            f"Sum of Zipf items probabilities"
            f" after normalization: {np.sum(zipf_probs_norm)}"
        )

        info("Zipf probabilities calculated")

        return zipf_probs_norm
    except (TypeError, ZeroDivisionError, ValueError) as e:
        msg = "Failed to calculate Zipf probabilities"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
