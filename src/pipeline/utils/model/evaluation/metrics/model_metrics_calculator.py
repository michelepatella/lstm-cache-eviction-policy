from typing import Dict, List

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from const import (
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE,
    MODEL_METRICS_CONFUSION_MATRIX,
    MODEL_METRICS_TOP_K_ACCURACY,
)
from pipeline.config.classes.Config import Config
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.model.evaluation.metrics.kappa_statistic_calculator import (
    calculate_cohen_kappa_score,
)
from pipeline.utils.model.evaluation.metrics.top_k_calculator import (
    calculate_top_k_accuracy,
)


def compute_model_metrics(
    targets: List[int],
    predictions: List[int],
    outputs: torch.Tensor,
    config: Config,
) -> Dict[str, int | float]:
    """
    Compute evaluation metrics for a model.

    This function calculates multiple evaluation metrics for
    a model, including:
        - Class-wise precision, recall, f1-score (class report).
        - Top-k accuracy.
        - Confusion matrix.
        - Cohen’s kappa score.

    Parameters:
        targets (List[int]): Ground truth class labels.
        predictions (List[int]): Predicted class labels.
        outputs (torch.Tensor): Model outputs.
        config (Config): Configuration object.

    Returns:
        Dict[str, int | float]: Dictionary containing class report with precision,
                                recall, f1-score, top-k accuracy, confusion matrix, and
                                Cohen's kappa score.

    Raises:
        RuntimeError: If an error occurs while computing model metrics, e.g.,:
            * Failed to compute class report due to mismatched lengths or invalid inputs.
            * Failed to compute confusion matrix due to mismatched lengths or invalid inputs.
    """
    # Prepare configuration
    top_k = config.evaluation.top_k

    debug(f"Targets length: {len(targets)}")
    debug(f"Predictions length: {len(predictions)}")
    debug(f"Outputs shape: {outputs.shape}")
    debug(f"Configured top-k: {top_k}")

    try:
        # Compute class report
        class_report = classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    except (ValueError, TypeError) as e:
        msg = "Failed to compute class report"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Calculate top-k accuracy
    top_k_accuracy = calculate_top_k_accuracy(targets, outputs, top_k)

    try:
        # Get confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)

        debug(f"Confusion matrix shape: {conf_matrix.shape}")
    except (ValueError, TypeError) as e:
        msg = "Failed to get confusion matrix"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Calculate Cohen's kappa score
    cohen_kappa_score = calculate_cohen_kappa_score(targets, predictions)

    # Collect metrics in a dictionary
    metrics = {
        MODEL_METRICS_CLASS_REPORT_NAME: class_report,
        MODEL_METRICS_TOP_K_ACCURACY: top_k_accuracy,
        MODEL_METRICS_CONFUSION_MATRIX: conf_matrix.tolist(),
        MODEL_METRICS_COHEN_KAPPA_SCORE: cohen_kappa_score,
    }

    info("Model metrics computed")

    return metrics
