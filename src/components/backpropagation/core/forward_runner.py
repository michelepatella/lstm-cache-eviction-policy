import torch

from components.device.mover import (
    move_to_device,
)
from components.logs.levels.error_logger import error
from components.loss.calculator import calculate_loss


def compute_forward(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Compute a forward pass through the model.

    This function handles moving batch to the specified device, executing the
    model's forward pass, and optionally computing the loss using the provided
    criterion. It is intended to streamline a single forward computation during
    training or evaluation.

    Args:
        batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Model batch.
        model (torch.nn.Module): The PyTorch model to compute forward pass for.
        device (torch.device): Device on which to perform computations.
        criterion (torch.nn.Module | None): Loss function.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor]:
            - loss: Computed loss using the criterion
              (None if criterion is not provided).
            - outputs: Model outputs from the forward pass.

    Raises:
    RuntimeError: If forward pass fails:
        * Batch unpacking fails because the provided batch tuple is
          malformed or has invalid types (ValueError, TypeError).
        * Model forward pass fails due to invalid input shapes or runtime
          errors in the model (RuntimeError).
    """
    try:
        # Unpack batch
        first_input, second_input, target = batch

        # Move batch to device
        first_input = move_to_device(first_input, device)
        second_input = move_to_device(second_input, device)
        target = move_to_device(target, device)

        # Compute forward pass
        outputs = model(first_input, second_input)

        # Calculate loss
        loss = calculate_loss(outputs, target, criterion)

        return loss, outputs
    except (ValueError, TypeError, RuntimeError) as e:
        msg = "Forward pass failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "batch_len": len(batch) if batch is not None else None,
                "batch_types": (
                    [type(x).__name__ for x in batch]
                    if batch is not None
                    else None
                ),
                "batch_shapes": (
                    [tuple(x.shape) for x in batch]
                    if batch is not None
                    else None
                ),
                "model": type(model).__name__ if model else None,
                "device": str(device),
                "criterion": type(criterion).__name__ if criterion else None,
                "context": "Model forward pass",
            },
        )
        raise RuntimeError(msg) from e
