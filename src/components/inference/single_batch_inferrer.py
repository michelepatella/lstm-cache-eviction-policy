import torch

from components.backpropagation.mc_dropout.forward_runner import (
    compute_mc_dropout_forward,
)
from components.device.mover import move_to_device
from components.logs.levels.error_logger import error
from components.loss.calculator import calculate_loss


def infer_single_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    num_features: int,
) -> tuple[
    float,
    list[int],
    list[int],
    list[torch.Tensor],
    list[torch.Tensor] | None,
]:
    """Perform inference on a single batch.

    This function processes one batch of data, moves tensors to
    the specified device, performs MC Dropout forward through the
    model, computes the batch loss, and extracts predictions,
    targets, outputs, and optionally variances.

    Args:
        batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            The batch to process.
        model (torch.nn.Module): PyTorch model to perform inference with.
        criterion (torch.nn.Module): Loss function for computing batch loss.
        device (torch.device): Device to run computations on.
        num_features (int): Number of features for the model.

    Returns:
        tuple[
            float,
            list[int],
            list[int],
            list[torch.Tensor],
            list[torch.Tensor] | None,
        ]:
            - loss: Float representing the computed loss for the batch.
            - predictions: List of predicted class indices for each
                           sample in the batch.
            - targets_list: List of ground truth labels corresponding to
                            each sample in the batch.
            - outputs_list: List of tensors containing model outputs for
                            each sample in the batch.
            - variances: List of tensors containing MC Dropout variances
                         for each sample (None if not computed).

    Raises:
        RuntimeError: If single batch inference fails:
            * Extracting the target fails because the batch is not a tuple/list
              of tensors or is empty (TypeError, IndexError).
            * Computing loss fails because outputs or target tensors are
              incompatible with the criterion (RuntimeError).
            * Preparing return data fails due to invalid tensor shapes, types,
              or operations (RuntimeError, TypeError, AttributeError,
              IndexError).
    """
    try:
        # Move entire batch to device
        batch = tuple(move_to_device(t, device) for t in batch)

        # Perform MC Dropout forward
        outputs_mean, outputs_var = compute_mc_dropout_forward(
            model,
            batch,
            device,
            num_features,
        )

        # Extract target from batch
        target = batch[-1]

        # Compute batch loss
        loss = calculate_loss(outputs_mean, target, criterion).item()

        # Prepare data to be returned
        predictions = torch.argmax(outputs_mean, dim=1).cpu().numpy().tolist()
        targets = target.cpu().numpy().tolist()
        outputs = [t.cpu() for t in outputs_mean]
        variances = (
            [t.cpu() for t in outputs_var] if outputs_var is not None else []
        )

        return loss, predictions, targets, outputs, variances
    except (TypeError, IndexError, AttributeError, RuntimeError) as e:
        msg = "Single batch inference failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model_name": model.__class__.__name__,
                "device": str(device),
                "features_num": num_features,
                "batch_type": str(type(batch)),
                "batch_length": (
                    len(batch) if hasattr(batch, "__len__") else None
                ),
                "context": "Single batch inference",
            },
        )
        raise RuntimeError(msg) from e
