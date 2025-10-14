from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pipeline.const import MODEL_COMPUTE_METRICS_DEFAULT
from utils.evaluation.metrics.utils.saver import save_model_metrics
from utils.inference.batches_inferrer import infer_batches
from utils.math.avg_calculator import calculate_average
from utils.evaluation.metrics.calculator import (
    calculate_model_metrics,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_features: int,
    top_k: int = None,
    model_results_save_path: str = None,
    compute_metrics: bool = MODEL_COMPUTE_METRICS_DEFAULT,
) -> Tuple[
    float, Dict[str, int | float] | None, List[Tensor], List[int], List[Tensor]
]:
    """
    Evaluate a model on a given data loader.

    This function performs inference over the provided data loader using
    the given model, computes the average loss, and optionally calculates
    evaluation metrics including classification report, top-k accuracy,
    and Cohen’s kappa score.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which to perform computations.
        num_features (int): Number of features for the model.
        top_k (int): Top-k for accuracy computation.
        model_results_save_path (str): Path to save metrics.
        compute_metrics (bool): Whether to compute evaluation metrics
                                in addition to loss.

    Returns:
        Tuple[
        float, Dict[str, int | float] | None,
        torch.Tensor, List[int], List[float]
        ]:
            Tuple containing the average loss,
            (optionally) a dictionary of
            evaluation metrics, model outputs,
            ground truth labels, and
            variances from MC dropout (if applicable).

    Raises:
        RuntimeError: If an error occurs during the
                      calculation of average loss, e.g.:
            * Division by zero if the data loader is empty.
            * Invalid type or attribute error when
              accessing the data loader length.
    """
    # Perform inference
    (
        total_loss,
        all_predictions,
        all_targets,
        all_outputs,
        all_variances,
    ) = infer_batches(model, data_loader, criterion, device, num_features)

    # Calculate average loss
    avg_loss = calculate_average([total_loss / len(data_loader)])

    debug(f"Average loss: {avg_loss}")

    # Compute metrics if requested
    metrics = None
    if compute_metrics:
        metrics = calculate_model_metrics(
            all_targets,
            all_predictions,
            all_outputs,
            top_k,
        )

        # Save metrics if requested (i.e., if the
        # model results save path is specified)
        if model_results_save_path is not None:
            save_model_metrics(metrics, avg_loss, model_results_save_path)

    info("Model evaluation completed")

    return avg_loss, metrics, all_outputs, all_targets, all_variances
