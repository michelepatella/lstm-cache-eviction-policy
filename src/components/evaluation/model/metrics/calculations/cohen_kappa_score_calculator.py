from typing import List

from sklearn.metrics import cohen_kappa_score

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_cohen_kappa_score(
    targets: List[int], predictions: List[int]
) -> float:
    """
    Calculate Cohen's kappa score.

    This function computes the Cohen's kappa score between the
    provided true labels and predicted ones, which measures
    the agreement between two raters while correcting for chance.

    Args:
        targets (List[int]): Ground truth labels.
        predictions (List[int]): Predicted labels.

    Returns:
        float: Cohen's kappa score.

    Raises:
        RuntimeError: If cohen kappa score calculation fails:
            * Invalid target or prediction types (TypeError).
            * Mismatched lengths or invalid values (ValueError).
    """
    try:
        debug(f"Targets length for Cohen's kappa score: {len(targets)}")
        debug(
            f"Predictions length for Cohen's kappa score: {len(predictions)}"
        )

        # Calculate Cohen's kappa score
        kappa = cohen_kappa_score(targets, predictions)

        info(f"Cohen's kappa score: {kappa}")

        return kappa
    except (ValueError, TypeError) as e:
        msg = "Failed to calculate Cohen's kappa score"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
