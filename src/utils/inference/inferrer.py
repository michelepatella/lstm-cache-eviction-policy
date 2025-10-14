from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from pipeline.config import Config
from pipeline.const import MC_DROPOUT_NUM_SAMPLES_DEFAULT
from utils.backpropagation.loss_calculator import calculate_loss
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.mc.forward_runner import (
    mc_forward_passes,
)
from utils.model.initialization.components.device_mover import (
    move_to_device,
)


def infer_batch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Config,
) -> Tuple[
    float, List[int], List[int], List[torch.Tensor], List[torch.Tensor]
]:
    """
    Perform inference on a data loader using MC Dropout forward passes.

    This function iterates over the data loader, computes model outputs
    using multiple stochastic forward passes (MC Dropout), calculates the
    loss per batch, and accumulates predictions, targets, outputs, and
    variances.

    Args:
        model (torch.nn.Module): The model to perform inference with.
        data_loader (DataLoader): DataLoader providing batches of data.
        criterion (torch.nn.Module): Loss function for computing batch loss.
        device (torch.device): Device on which to perform computation.
        config (Config): Configuration object.

    Returns:
        Tuple[
        float, List[int], List[int],
        List[torch.Tensor], List[torch.Tensor
        ]]:
            Tuple containing sum of batch losses,
            predicted class indices per sample,
            ground truth labels per sample, model
            outputs per batch, and variances
            from MC dropout.

    Raises:
        RuntimeError: If an error occurs during batch processing, e.g.:
            * Batch unpacking fails.
            * Moving tensors to device fails.
            * Forward pass or loss computation fails.
            * Argmax or conversion to NumPy fails.
    """
    # Prepare configuration
    num_features = config.model.general.features

    # Initialization
    total_loss = 0.0
    all_predictions: List[int] = []
    all_targets: List[int] = []
    all_outputs: List[torch.Tensor] = []
    all_vars: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

                # Unpack batch
                features, keys, targets = batch

                debug(
                    f"Processing batch {batch_idx}: "
                    f"features shape: {features.shape}, "
                    f"keys shape: {keys.shape},"
                    f" targets shape: {targets.shape}"
                )

                # Move tensors to device
                features, keys, targets = (
                    move_to_device(features, device),
                    move_to_device(keys, device),
                    move_to_device(targets, device),
                )

                # Perform MC Dropout forward passes
                outputs_mean, outputs_var, _ = mc_forward_passes(
                    model,
                    (features, keys, targets),
                    device,
                    num_features,
                    MC_DROPOUT_NUM_SAMPLES_DEFAULT,
                )

                debug(f"Outputs mean shape: {outputs_mean.shape}")

                # Collect variance if available
                if outputs_var is not None:
                    all_vars.extend(outputs_var.cpu())
                    debug(f"Outputs variance shape: {outputs_var.shape}")

                # Compute batch loss
                loss = calculate_loss(outputs_mean, targets, criterion)
                total_loss += loss.item()

                debug(f"Batch {batch_idx} loss: {loss.item():.4f}")

                # Store predictions, targets, and outputs
                predictions = torch.argmax(outputs_mean, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs_mean.cpu())
            except (IndexError, ValueError, RuntimeError, TypeError) as e:
                msg = "Failed to infer batch"
                error("%s: %s", msg, e)
                raise RuntimeError(msg) from e

    info("Batch inference completed")

    return total_loss, all_predictions, all_targets, all_outputs, all_vars
