from typing import Dict, List

import torch

from const import (
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE_NAME,
    MODEL_METRICS_TOP_K_ACCURACY_NAME,
)
from calculations.evaluation.model.metrics.calculations.class_report_generator import (
    generate_class_report,
)
from calculations.evaluation.model.metrics.calculations.cohen_kappa_score_calculator import (
    calculate_cohen_kappa_score,
)
from calculations.evaluation.model.metrics.calculations.top_k_accuracy_calculator import (
    calculate_top_k_accuracy,
)
from calculations.logs.levels.debug_logger import debug
from calculations.logs.levels.info_logger import info


def calculate_model_metrics(
    targets: List[int],
    predictions: List[int],
    outputs: List[torch.Tensor],
    top_k: int,
) -> Dict[str, int | float]:
    """
    Calculate evaluation metrics for a model.

    This function calculates multiple evaluation metrics for
    a model, including:
        - Class-wise precision, recall, f1-score (class report).
        - Top-k accuracy.
        - Cohen’s kappa score.

    Args:
        targets (List[int]): Ground truth class labels.
        predictions (List[int]): Predicted class labels.
        outputs (List[torch.Tensor]): Model outputs.
        top_k (int): Top-k to be considered for accuracy calculation.

    Returns:
        Dict[str, int | float]: Dictionary containing class
                                report with precision,
                                recall, f1-score, top-k accuracy,
                                and Cohen's kappa score.

    Raises:
        RuntimeError: If an error occurs while
                      computing model metrics, e.g.:
            * Failed to compute class report due to
              mismatched lengths or invalid inputs.
    """
    debug(f"Targets length: {len(targets)}")
    debug(f"Predictions length: {len(predictions)}")
    debug(f"Outputs length: {len(outputs)}")
    debug(f"Top-k: {top_k}")

    # Generate a class report
    class_report = generate_class_report(targets, predictions)

    # Calculate top-k accuracy
    top_k_accuracy = calculate_top_k_accuracy(targets, outputs, top_k)

    # Calculate Cohen's kappa score
    cohen_kappa_score = calculate_cohen_kappa_score(targets, predictions)

    # Collect metrics in a dictionary
    metrics = {
        MODEL_METRICS_CLASS_REPORT_NAME: class_report,
        MODEL_METRICS_TOP_K_ACCURACY_NAME: top_k_accuracy,
        MODEL_METRICS_COHEN_KAPPA_SCORE_NAME: cohen_kappa_score,
    }

    info("Model metrics calculated")

    return metrics
