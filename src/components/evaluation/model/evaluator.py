"""evaluator.py

Utility module for evaluating PyTorch models.

This module provides the `evaluate_model` function, which performs
inference on a model over a given dataset, computes average loss,
optionally calculates evaluation metrics, and can save the results.

Functions:
    evaluate_model(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        num_workers: int,
        model_results_save_path: str = None,
        compute_metrics: bool = True
    ) -> tuple[
        float,
        dict[str, int | float] | None,
        list[Tensor],
        list[int],
        list[Tensor]
    ]
        Performs model evaluation, returning average loss, optional metrics,
        model outputs, ground truth targets, and variances if applicable.
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from components.const import MODEL_COMPUTE_METRICS_DEFAULT
from components.evaluation.model.metrics.calculator import (
    calculate_model_metrics,
)
from components.evaluation.model.metrics.io.saver import save_model_metrics
from components.inference.batches_inferrer import infer_batches
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_workers: int,
    model_results_save_path: str = None,
    compute_metrics: bool = MODEL_COMPUTE_METRICS_DEFAULT,
) -> tuple[
    float,
    dict[str, int | float] | None,
    list[Tensor],
    list[int],
    list[Tensor],
]:
    """Evaluate a model on a given data loader.

    This function performs inference over the provided data loader using
    the given model, computes the average loss, and optionally calculates
    evaluation metrics including classification report, top-k accuracy,
    and Cohen’s kappa score.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which to perform computations.
        num_workers (int): Number of workers to use during inference.
        model_results_save_path (str): Path to save metrics.
        compute_metrics (bool): Whether to compute evaluation metrics
                                in addition to loss.

    Returns:
        tuple[
            float,
            dict[str, int | float] | None,
            list[Tensor],
            list[int],
            list[Tensor],
        ]:
            - avg_loss: Float representing the average loss over the data
                        loader.
            - metrics: Optional dictionary of evaluation metrics
                       (classification report, top-k accuracy, Cohen’s
                       kappa), or None if not computed.
            - all_outputs: List of tensors containing the model outputs
                           per batch.
            - all_targets: List of ground truth labels corresponding to
                           the inputs.
            - all_variances: List of tensors containing variances from MC
                             dropout (if applicable), otherwise empty.
    """
    debug(
        "Model evaluation started",
        extra={
            "device": str(device),
            "compute_metrics": compute_metrics,
            "model_type": type(model).__name__,
            "context": "Model evaluation",
        },
    )

    # Perform inference
    (
        total_loss,
        all_predictions,
        all_targets,
        all_outputs,
        all_variances,
    ) = infer_batches(model, data_loader, criterion, device, num_workers)

    # Calculate average loss
    avg_loss = calculate_average([total_loss / len(data_loader)])

    info(
        "Average loss computed",
        extra={
            "loss_avg": None
            if np.isinf(avg_loss) or np.isnan(avg_loss)
            else float(avg_loss),
            "batches_num": len(data_loader),
            "outputs_num": len(all_outputs),
            "targets_num": len(all_targets),
            "context": "Model evaluation",
        },
    )

    # Compute metrics if requested
    metrics = None
    if compute_metrics:
        metrics = calculate_model_metrics(
            all_targets,
            all_predictions,
            all_outputs,
        )

        # Save metrics if requested (i.e., if the
        # model results save path is specified)
        if model_results_save_path is not None:
            save_model_metrics(metrics, avg_loss, model_results_save_path)

    debug(
        "Model evaluation completed",
        extra={
            "loss_avg": None
            if np.isinf(avg_loss) or np.isnan(avg_loss)
            else float(avg_loss),
            "metrics_computed": metrics is not None,
            "batches_num": len(data_loader),
            "outputs_num": len(all_outputs),
            "targets_num": len(all_targets),
            "variances_num": len(all_variances),
            "context": "Model evaluation",
        },
    )

    return avg_loss, metrics, all_outputs, all_targets, all_variances
