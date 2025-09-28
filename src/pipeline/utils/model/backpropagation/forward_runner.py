from typing import Optional, Tuple

import torch
from torch import nn

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info


def compute_forward(
    inputs,
    model: nn.Module,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Compute a forward pass through the model.

    This function handles both training and inference:
        - If `inputs` is a tuple of length `num_features + 1`, it unpacks
          features, keys, and target, moves them to the specified device,
          and computes model outputs.
        - If `criterion` is provided, computes the loss using the model outputs
          and target.

    Parameters:
        inputs: Model inputs, either a tuple including the target or inputs ready for model.
        model (nn.Module): The PyTorch model to evaluate.
        criterion (Optional[nn.Module]): Loss function. If None, loss is not computed.
        device (torch.device): Device on which to perform computations.

    Returns:
        Tuple[Optional[torch.Tensor], torch.Tensor]:
            - loss: Computed loss (None if criterion is None).
            - outputs: Model outputs.
    """
    # Check if inputs include target
    if isinstance(inputs, tuple) and len(inputs) == model.num_features + 1:
        x_features, x_keys, y_key = inputs

        # Move to device
        x_features = x_features.to(device)
        x_keys = x_keys.to(device)
        y_key = y_key.to(device)

        # Forward pass
        outputs = model(x_features, x_keys)

        debug(f"Input features shape: {x_features.shape}")
        debug(f"Input keys shape: {x_keys.shape}")
        debug(f"Target batch shape: {y_key.shape}")
        debug(f"Model output shape: {outputs.shape}")

    else:
        # Assume inputs are directly compatible with model
        outputs = model(*inputs)
        debug(f"Model output shape: {outputs.shape}")

    # Compute loss if criterion provided
    loss = None
    if criterion is not None:
        try:
            loss = criterion(outputs, y_key)
            debug(f"Loss computed: {loss.item()}")
        except Exception as e:
            raise RuntimeError(f"Failed to compute loss: {e}") from e

    info("Forward pass completed")

    return loss, outputs
