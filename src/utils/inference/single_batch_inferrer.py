from typing import List, Tuple

import torch

from utils.backpropagation.mc_dropout.runner import (
    compute_mc_dropout_forward_passes,
)
from utils.device.mover import move_to_device
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.loss_calculator import calculate_loss


def infer_single_batch(
    batch_idx: int,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    num_features: int,
) -> Tuple[
    float, List[int], List[int], List[torch.Tensor], List[torch.Tensor]
]:
    """
    Perform inference on a single batch.

    This function processes one batch of data, moves tensors to the specified device,
    performs MC Dropout forward passes through the model, computes the batch loss,
    and extracts predictions, targets, outputs, and optionally variances.

    Args:
        batch_idx (int): Index of the current batch.
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            The batch to process (features, keys, targets).
        model (torch.nn.Module): Model to perform inference with.
        criterion (torch.nn.Module): Loss function for computing batch loss.
        device (torch.device): Device to run computations on.
        num_features (int): Number of features for the model.

    Returns:
        Tuple[
            float, List[int], List[int], List[torch.Tensor], List[torch.Tensor]
        ]: Tuple containing loss, predictions, targets, outputs, and optionally variances.

    Raises:
        RuntimeError: If an error occurs during batch processing, e.g.:
            * Batch unpacking fails.
            * Moving tensors to device fails.
            * Forward pass or loss computation fails.
            * Argmax or conversion to NumPy fails.
    """
    try:
        # Unpack batch
        features, keys, targets = batch

        debug(
            f"Processing batch {batch_idx}: "
            f"features shape: {features.shape}, "
            f"keys shape: {keys.shape}, "
            f"targets shape: {targets.shape}"
        )

        # Move tensors to device
        features, keys, targets = (
            move_to_device(features, device),
            move_to_device(keys, device),
            move_to_device(targets, device),
        )

        # Perform MC Dropout forward passes
        outputs_mean, outputs_var, _ = compute_mc_dropout_forward_passes(
            model, (features, keys, targets), device, num_features
        )

        debug(f"Outputs mean shape: {outputs_mean.shape}")

        # Compute batch loss
        loss = calculate_loss(outputs_mean, targets, criterion).item()

        # Collect data to be returned
        predictions = torch.argmax(outputs_mean, dim=1).cpu().numpy().tolist()
        targets_list = targets.cpu().numpy().tolist()
        outputs_list = [t.cpu() for t in outputs_mean]
        variances = (
            [t.cpu() for t in outputs_var] if outputs_var is not None else []
        )

        info("Single batch inference completed")

        return loss, predictions, targets_list, outputs_list, variances
    except (IndexError, ValueError, RuntimeError, TypeError) as e:
        msg = "Failed to infer single batch"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
