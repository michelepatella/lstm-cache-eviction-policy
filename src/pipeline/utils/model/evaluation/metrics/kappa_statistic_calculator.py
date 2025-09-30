from typing import List

from sklearn.metrics import cohen_kappa_score

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def calculate_cohen_kappa_score(
    targets: List[int], predictions: List[int]
) -> float:
    """
    Calculate Cohen's kappa score.

    This function computes the Cohen's kappa score between the
    provided true labels and predicted ones, which measures
    the agreement between two raters while correcting for chance.

    Parameters:
        targets (List[int]): Ground truth labels.
        predictions (List[int]): Predicted labels.

    Returns:
        float: Cohen's kappa score.

    Raises:
        RuntimeError: If an error occurs while calculating Cohen's kappa score, e.g.,:
            * Targets and predictions have different lengths.
            * Targets and/or predictions have not compatible type with
              Cohen's kappa score calculation.
    """
    debug(f"Targets length for Cohen's kappa score: {len(targets)}")
    debug(f"Predictions length for Cohen's kappa score: {len(predictions)}")

    try:
        # Calculate Cohen's kappa score
        kappa = cohen_kappa_score(targets, predictions)
    except (ValueError, TypeError) as e:
        msg = "Failed to calculate Cohen's kappa score"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Cohen's kappa score calculated")

    return kappa
