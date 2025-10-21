from typing import Dict, List

from sklearn.metrics import classification_report

from components.const import (
    MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT_ENABLED,
    MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_class_report(
    targets: List[int],
    predictions: List[int],
    output_dict: bool = MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT_ENABLED,
    zero_division: int = MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION,
) -> Dict[str, float]:
    """
    Calculate a classification report.

    This function computes the classification report given the
    true targets and predicted labels.

    Args:
        targets (List[int]): True class labels.
        predictions (List[int]): Predicted class labels.
        output_dict (bool): If True, return results as a dictionary.
        zero_division (int): Value to use when a zero division occurs.

    Returns:
        Dict[str, float]: Classification metrics per class.

    Raises:
        RuntimeError: If classification report calculation fails due to:
            * Invalid target or prediction types (TypeError).
            * Inconsistent lengths or values (ValueError).
    """
    try:
        # Generate classification report
        class_report = classification_report(
            targets,
            predictions,
            output_dict=output_dict,
            zero_division=zero_division,
        )

        info(f"Classification report: {class_report}")

        return class_report
    except (ValueError, TypeError) as e:
        msg = "Failed to compute classification report"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
