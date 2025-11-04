"""model_metrics_calculator.py

Module responsible for aggregating and calculating various
evaluation metrics for the model.

This module uses ground truth labels, predicted labels, and raw
model outputs to compute a comprehensive set of metrics, including
the classification report (precision, recall, f1-score) and Cohen's
Kappa score.

Functions:
    calculate_model_metrics(
        targets: list[int],
        predictions: list[int],
        outputs: list[torch.Tensor]
    ) -> dict[str, int | float]
        Aggregates and computes the complete set of evaluation metrics
        for a model's performance.
"""

import torch

from components.const import (
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE_NAME,
)
from components.evaluation.model.metrics.calculations.class_report_calculator import (
    calculate_class_report,
)
from components.evaluation.model.metrics.calculations.cohen_kappa_score_calculator import (
    calculate_cohen_kappa_score,
)
from components.logs.levels.debug_logger import debug


def calculate_model_metrics(
    targets: list[int],
    predictions: list[int],
    outputs: list[torch.Tensor],
) -> dict[str, int | float]:
    """Calculate evaluation metrics for a model.

    This function calculates multiple evaluation metrics for
    a model, including:
        - Class-wise precision, recall, f1-score (class report).
        - Top-k accuracy.
        - Cohen’s kappa score.

    Args:
        targets (list[int]): Ground truth class labels.
        predictions (list[int]): Predicted class labels.
        outputs (list[torch.Tensor]): Model outputs.

    Returns:
        dict[str, int | float]: Dictionary containing class report with
                                precision, recall, f1-score, top-k
                                accuracy, and Cohen's kappa score.
    """
    debug(
        "Model metrics calculation started",
        extra={
            "targets_num": len(targets),
            "predictions_num": len(predictions),
            "outputs_num": len(outputs),
            "context": "Model metrics calculation",
        },
    )

    # Generate a class report
    class_report = calculate_class_report(targets, predictions)

    # Calculate Cohen's kappa score
    cohen_kappa_score = calculate_cohen_kappa_score(targets, predictions)

    # Collect metrics in a dictionary
    metrics = {
        MODEL_METRICS_CLASS_REPORT_NAME: class_report,
        MODEL_METRICS_COHEN_KAPPA_SCORE_NAME: cohen_kappa_score,
    }

    debug(
        "Model metrics calculation completed",
        extra={
            "targets_num": len(targets),
            "predictions_num": len(predictions),
            "outputs_num": len(outputs),
            "context": "Model metrics calculation",
        },
    )

    return metrics
