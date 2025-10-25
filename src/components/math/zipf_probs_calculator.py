import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
            "Zipf probability calculation started",
            extra={
                "items_shape": (
                    items.shape if hasattr(items, "shape") else None
                ),
                "items_dtype": (
                    str(items.dtype) if hasattr(items, "dtype") else None
                ),
                "alpha": alpha,
                "items_num": len(items) if hasattr(items, "__len__") else None,
                "context": "Zipf probability calculation",
            },
        )
        # Calculate the probability of the items
        # according to the Zipf's distribution formula
        zipf_probs = 1.0 / np.power(items, alpha)
        # Normalize probabilities to make sum to 1
        zipf_probs_norm = zipf_probs / np.sum(zipf_probs)

        debug(
            "Zipf probability calculation completed",
            extra={
                "alpha": alpha,
                "probs_sum": np.sum(zipf_probs_norm),
                "prob_max": np.max(zipf_probs_norm),
                "prob_min": np.min(zipf_probs_norm),
                "items_num": len(zipf_probs_norm),
                "context": "Zipf probability calculation",
            },
        )

        return zipf_probs_norm
    except (TypeError, ZeroDivisionError, ValueError) as e:
        msg = "Zipf probability calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "items_type": type(items).__name__,
                "items_shape": (
                    items.shape if hasattr(items, "shape") else None
                ),
                "alpha": alpha,
                "context": "Zipf probability calculation",
            },
        )
        raise RuntimeError(msg) from e
