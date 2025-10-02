from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from const import COMPUTE_METRICS_DEFAULT
from pipeline.config.classes.Config import Config
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.model.evaluation.inference import infer_batch
from pipeline.utils.model.evaluation.metrics.model_metrics_calculator import (
    compute_model_metrics,
)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Config,
    compute_metrics: bool = COMPUTE_METRICS_DEFAULT,
) -> Tuple[
    float, Dict[str, int | float] | None, List[Tensor], List[int], List[Tensor]
]:
    """
    Evaluate a model on a given data loader.

    This function performs inference over the provided data loader using
    the given model, computes the average loss, and optionally calculates
    evaluation metrics including classification report, top-k accuracy,
    confusion matrix, and Cohen’s kappa score.

    Parameters:
        model (torch.nn.Module): Model to evaluate.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which to perform computations.
        config (Config): Configuration object.
        compute_metrics (bool): Whether to compute evaluation metrics
                                in addition to loss.

    Returns:
        Tuple[float, Dict[str, int | float] | None, torch.Tensor, List[int], List[float]]:
            Tuple containing the average loss, (optionally) a dictionary of
            evaluation metrics, model outputs, ground truth labels, and
            variances from MC dropout (if applicable).

    Raises:
        RuntimeError: If an error occurs during the calculation of average loss, e.g.:
            * Division by zero if the data loader is empty.
            * Invalid type or attribute error when accessing the data loader length.
    """
    # Perform inference
    (
        total_loss,
        all_predictions,
        all_targets,
        all_outputs,
        all_vars,
    ) = infer_batch(model, data_loader, criterion, device, config)

    try:
        debug(f"Total loss accumulated: {total_loss:.4f}")
        debug(f"Number of batches: {len(data_loader)}")

        # Calculate average loss
        avg_loss = total_loss / len(data_loader)

        debug(f"Average loss: {avg_loss:.4f}")
    except (ZeroDivisionError, TypeError, AttributeError) as e:
        msg = "Failed to compute average loss during model evaluation"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Compute metrics if requested
    metrics = None
    if compute_metrics:
        metrics = compute_model_metrics(
            all_targets,
            all_predictions,
            all_outputs,
            config,
        )
        debug("Evaluation metrics computed")

    info("Model evaluation completed")

    return avg_loss, metrics, all_outputs, all_targets, all_vars
