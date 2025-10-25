from typing import Dict, List, Union

import torch

from components.const import (
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE_NAME,
    MODEL_METRICS_TOP_K_ACCURACY_NAME,
)
from components.evaluation.model.metrics.calculations.class_report_calculator import (
    calculate_class_report,
)
from components.evaluation.model.metrics.calculations.cohen_kappa_score_calculator import (
    calculate_cohen_kappa_score,
)
from components.evaluation.model.metrics.calculations.top_k_accuracy_calculator import (
    calculate_top_k_accuracy,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info


def calculate_model_metrics(
    targets: List[int],
    predictions: List[int],
    outputs: List[torch.Tensor],
    top_k: int,
) -> Dict[str, Union[int, float]]:
    """
    Calculate evaluation metrics for a model.

    This function calculates multiple evaluation metrics for a model, including:
        - Class-wise precision, recall, f1-score (class report).
        - Top-k accuracy.
        - Cohen’s kappa score.

    Args:
        targets (List[int]): Ground truth class labels.
        predictions (List[int]): Predicted class labels.
        outputs (List[torch.Tensor]): Model outputs.
        top_k (int): Top-k to be considered for accuracy calculation.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing class report with precision,
                                      recall, f1-score, top-k accuracy, and Cohen's kappa
                                      score.
    """
    debug(
        "Model metrics calculation started",
        extra={
            "targets_num": len(targets),
            "predictions_num": len(predictions),
            "outputs_num": len(outputs),
            "top_k": top_k,
            "context": "Model metrics calculation",
        },
    )

    # Generate a class report
    class_report = calculate_class_report(targets, predictions)

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

    debug(
        "Model metrics calculation completed",
        extra={
            "metrics": metrics,
            "targets_num": len(targets),
            "predictions_num": len(predictions),
            "outputs_num": len(outputs),
            "top_k": top_k,
            "context": "Model metrics calculation",
        },
    )

    return metrics
