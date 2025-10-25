from typing import List

from sklearn.metrics import cohen_kappa_score

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
        # Calculate Cohen's kappa score
        kappa = cohen_kappa_score(targets, predictions)

        info(
            "Cohen's kappa score calculated",
            extra={
                "num_targets": len(targets),
                "num_predictions": len(predictions),
                "cohen_kappa_score": kappa,
                "context": "Cohen's kappa calculation",
            },
        )

        return kappa
    except (ValueError, TypeError) as e:
        msg = "Cohen's kappa score calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "num_targets": len(targets) if targets else 0,
                "num_predictions": len(predictions) if predictions else 0,
                "context": "Cohen's kappa calculation",
            },
        )
        raise RuntimeError(msg) from e
