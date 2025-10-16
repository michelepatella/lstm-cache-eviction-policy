from typing import Tuple, Union

import torch

from components.loss.calculator import calculate_loss
from components.device.mover import (
    move_to_device,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info


def compute_forward(
    inputs: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    model: torch.nn.Module,
    criterion: torch.nn.Module | None,
    device: torch.device,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """
    Compute a forward pass through the model.

    This function handles both training and inference:
        - If the input is a tuple of length
          equals number of features plus target,
          it unpacks features, keys, and target,
          moves them to the specified device,
          and computes model outputs.
        - If a criterion is provided, computes the
          loss using the model outputs and target.

    Args:
        inputs (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]): Model inputs, either a tuple including the
            target or inputs ready for model.
        model (torch.nn.Module): The PyTorch model to compute
                                 forward pass for.
        criterion (torch.nn.Module | None): Loss function.
                                            If None, loss is not computed.
        device (torch.device): Device on which to perform
                               computations.

    Returns:
        Tuple[torch.Tensor | None, torch.Tensor]:
            - loss: Computed loss using the criterion (None if criterion is not provided).
            - outputs: Model outputs from the forward pass.

    Raises:
        RuntimeError: If an error occurs while calculating the loss.
    """
    # Unpack inputs
    x_features, x_keys, y_key = inputs

    debug(f"Input features shape: {x_features.shape}")
    debug(f"Input keys shape: {x_keys.shape}")
    debug(f"Target batch shape: {y_key.shape}")

    # Move to device
    x_features = move_to_device(x_features, device)
    x_keys = move_to_device(x_keys, device)
    y_key = move_to_device(y_key, device)

    debug(f"Inputs moved to device: {device}")

    # Forward pass
    outputs = model(x_features, x_keys)

    debug(f"Model output shape: {outputs.shape}")

    # Calculate loss
    loss = calculate_loss(outputs, y_key, criterion)

    info("Forward pass completed")

    return loss, outputs
