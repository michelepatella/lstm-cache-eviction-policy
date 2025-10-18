from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from components.inference.single_batch_inferrer import infer_single_batch
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def infer_batches(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_features: int,
) -> Tuple[
    float, List[int], List[int], List[torch.Tensor], List[torch.Tensor]
]:
    """
    Perform inference on a data loader over several batches.

    This function iterates over the data loader, infer all batches,
    and accumulates batch losses, predictions, targets, outputs,
    and variances.

    Args:
        model (torch.nn.Module): The PyTorch model to perform inference with.
        data_loader (DataLoader): DataLoader providing batches of data.
        criterion (torch.nn.Module): Loss function for computing batch loss.
        device (torch.device): Device on which to perform computation.
        num_features (int): Number of features for the model.

    Returns:
        Tuple[
            float, List[int], List[int], List[torch.Tensor], List[torch.Tensor]
        ]:
            - total_loss: Float representing the sum of batch losses across
                          the data loader.
            - all_predictions: List of predicted class indices for each sample.
            - all_targets: List of ground truth labels corresponding to each sample.
            - all_outputs: List of tensors containing model outputs per batch.
            - all_variances: List of tensors containing variances from MC dropout
                             per batch (if applicable).

    Raises:
        RuntimeError: If batches inference fails:
            * If the dataloader is not iterable (TypeError).
            * If any of the batch results cannot be added or extended due to
              incorrect types (TypeError).
    """
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_outputs = []
    all_variances = []

    with torch.no_grad():
        # Iterate over all the batches of the
        # data loader
        try:
            for batch_idx, batch in enumerate(data_loader):
                # Infer the current batch
                (
                    loss,
                    predictions,
                    targets,
                    outputs,
                    variances,
                ) = infer_single_batch(
                    batch, model, criterion, device, num_features
                )

                # Keep track of results
                total_loss += loss
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_outputs.extend(outputs)
                all_variances.extend(variances)

                debug(f"Batch {batch_idx} inferred")
        except TypeError as e:
            msg = "Failed to infer batches"
            error("%s: %s", msg, e)

    info("Batches inference completed")

    return total_loss, all_predictions, all_targets, all_outputs, all_variances
