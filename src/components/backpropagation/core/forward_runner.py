from typing import Tuple, Optional, Union

import torch

from components.device.mover import (
    move_to_device,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.loss.calculator import calculate_loss


def compute_forward(
    batch: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ],
    model: torch.nn.Module,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Compute a forward pass through the model.

    This function handles moving batch to the specified device, executing the
    model's forward pass, and optionally computing the loss using the provided
    criterion. It is intended to streamline a single forward computation during
    training or evaluation.

    Args:
        batch (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]
        ]): Model batch, either a tuple including the target or inputs
            ready for model.
        model (torch.nn.Module): The PyTorch model to compute forward pass for.
        device (torch.device): Device on which to perform computations.
        criterion (Optional[torch.nn.Module]): Loss function.

    Returns:
        Tuple[Optional[torch.Tensor], torch.Tensor]:
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
        debug(f"First input shape: {first_input.shape}")
        debug(f"Second input shape: {second_input.shape}")
        debug(f"Target shape: {target.shape}")
    except (ValueError, TypeError) as e:
        msg = "Failed to unpack batch"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Move batch to device
    first_input = move_to_device(first_input, device)
    second_input = move_to_device(second_input, device)
    target = move_to_device(target, device)
    debug(f"Batch moved to device: {device}")

    try:
        # Compute forward pass
        outputs = model(first_input, second_input)
        debug(f"Model outputs shape: {outputs.shape}")
    except RuntimeError as e:
        msg = "Failed to compute forward pass"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Calculate loss
    loss = calculate_loss(outputs, target, criterion)

    info("Forward pass completed")

    return loss, outputs
