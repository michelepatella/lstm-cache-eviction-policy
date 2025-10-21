from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from components.evaluation.model.metrics.calculator import (
    calculate_model_metrics,
)
from components.evaluation.model.metrics.io.saver import save_model_metrics
from components.inference.batches_inferrer import infer_batches
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average
from const import MODEL_COMPUTE_METRICS_DEFAULT


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
    float,
    Optional[Dict[str, Union[int, float]]],
    List[Tensor],
    List[int],
    List[Tensor],
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
        compute_metrics (bool): Whether to compute evaluation metrics in addition to loss.

    Returns:
        Tuple[
            float, Optional[Dict[str, Union[int, float]]], List[Tensor],
            List[int], List[Tensor]
        ]:
            - avg_loss: Float representing the average loss over the data loader.
            - metrics: Optional dictionary of evaluation metrics (classification report,
                       top-k accuracy, Cohen’s kappa), or None if not computed.
            - all_outputs: List of tensors containing the model outputs per batch.
            - all_targets: List of ground truth labels corresponding to the inputs.
            - all_variances: List of tensors containing variances from MC dropout
                             (if applicable), otherwise empty.
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
    info(f"Average loss: {avg_loss}")

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

    debug("Model evaluation completed")

    return avg_loss, metrics, all_outputs, all_targets, all_variances
