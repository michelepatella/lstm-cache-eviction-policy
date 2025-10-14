from typing import Dict, List

from sklearn.metrics import classification_report

from pipeline.const import MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT, MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def generate_class_report(
    targets: List[int], predictions: List[int]
) -> Dict[str, float]:
    """
    Generate a classification report.

    This function computes the classification report given
    the true targets and predicted labels.

    Args:
        targets (List[int]): True class labels.
        predictions (List[int]): Predicted class labels.

    Returns:
        Dict[str, float]: Classification metrics per class.

    Raises:
        RuntimeError: If computation of the classification report fails.
    """
    try:
        # Generate classification report
        class_report = classification_report(
            targets,
            predictions,
            output_dict=MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT,
            zero_division=MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION,
        )
    except (ValueError, TypeError) as e:
        msg = "Failed to compute classification report"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Classification report generated")

    return class_report
