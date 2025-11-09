"""batches_inferrer.py

Utility module for performing inference over multiple data batches.

This module provides the `infer_batches` function, which iterates over
a PyTorch DataLoader, performs inference on each batch using a given model,
computes batch losses, and aggregates predictions, targets, outputs, and
optional variance tensors.

Functions:
    infer_batches(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        num_workers: int
    ) -> tuple[
        float,
        list[int],
        list[int],
        list[torch.Tensor],
        list[torch.Tensor]
    ]
        Performs batch-wise inference on the provided data loader and
        returns aggregated loss, predictions, targets, outputs, and
        variance tensors.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader

from components.inference.single_batch_inferrer import infer_single_batch
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def infer_batches(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_workers: int,
) -> tuple[
    float,
    list[int],
    list[int],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Perform inference on a data loader over several batches.

    This function iterates over the data loader, infer all batches,
    and accumulates batch losses, predictions, targets, outputs,
    and variances.

    Args:
        model (torch.nn.Module): The PyTorch model to perform inference with.
        data_loader (DataLoader): DataLoader providing batches of data.
        criterion (torch.nn.Module): Loss function for computing batch loss.
        device (torch.device): Device on which to perform computation.
        num_workers (int): Number of workers to parallelize inference.

    Returns:
    tuple[
        float,
        list[int],
        list[int],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
            - total_loss: Float representing the sum of batch losses across
                          the data loader.
            - all_predictions: List of predicted class indices for each sample.
            - all_targets: List of ground truth labels corresponding to
                           each sample.
            - all_outputs: List of tensors containing model outputs per batch.
            - all_variances: List of tensors containing variances from MC
                             dropout per batch (if applicable).

    Raises:
        RuntimeError: If batches inference fails:
            * If the dataloader is not iterable (TypeError).
            * If any of the batch results cannot be added or extended due to
              incorrect types (TypeError).
    """
    try:
        debug(
            "Batches inference started",
            extra={
                "model": type(model).__name__,
                "device": str(device),
                "batches_num": (
                    len(data_loader)
                    if hasattr(data_loader, "__len__")
                    else None
                ),
                "context": "Batches inference",
            },
        )

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_outputs = []
        all_variances = []

        # Run multi-thread inference over all the batches
        # according to the given number of workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    infer_single_batch,
                    batch,
                    model,
                    criterion,
                    device,
                )
                for batch in data_loader
            ]

            # Collect all the results
            for f in as_completed(futures):
                loss, preds, targets, outputs, variances = f.result()
                total_loss += loss
                all_predictions.extend(preds)
                all_targets.extend(targets)
                all_outputs.extend(outputs)
                all_variances.extend(variances)

        debug(
            "Batches inference completed",
            extra={
                "loss_tot": None
                if np.isinf(total_loss) or np.isnan(total_loss)
                else float(total_loss),
                "predictions_num": len(all_predictions),
                "targets_num": len(all_targets),
                "outputs_num": len(all_outputs),
                "variances_num": len(all_variances),
                "context": "Batches inference",
            },
        )

        return (
            total_loss,
            all_predictions,
            all_targets,
            all_outputs,
            all_variances,
        )
    except TypeError as e:
        msg = "Batches inference failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model": type(model).__name__,
                "device": str(device),
                "context": "Batches inference",
            },
        )
        raise RuntimeError(e) from e
