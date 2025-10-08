from typing import List

from numpy import ndarray
from sklearn.metrics import confusion_matrix

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def generate_confusion_matrix(
    targets: List[int], predictions: List[int]
) -> ndarray:
    """
    Compute the confusion matrix.

    This function generates the confusion matrix,
    given a set of targets and predictions.

    Args:
        targets (List[int]): True class labels.
        predictions (List[int]): Predicted class labels.

    Returns:
        ndarray: Confusion matrix of shape (n_classes, n_classes).

    Raises:
        RuntimeError: If an error occurs while computing
                      the confusion matrix fails, e.g.:
            * If targets and predictions lengths mismatch.
            * If input contains invalid values or types.
    """
    try:
        # Get confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)

        debug(f"Confusion matrix shape: {conf_matrix.shape}")
    except (ValueError, TypeError) as e:
        msg = "Failed to generate confusion matrix"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Confusion matrix generated")

    return conf_matrix
