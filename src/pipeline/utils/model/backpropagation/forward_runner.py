from typing import Tuple, Union

import torch
from torch import nn

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def compute_forward(
    inputs: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    model: nn.Module,
    criterion: nn.Module | None,
    device: torch.device,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """
    Compute a forward pass through the model.

    This function handles both training and inference:
        - If the input is a tuple of length equals number of features plus target,
          it unpacks features, keys, and target, moves them to the specified device,
          and computes model outputs.
        - If a criterion is provided, computes the loss using the model outputs
          and target.

    Parameters:
        inputs (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]): Model inputs, either a tuple including the target or inputs ready for model.
        model (nn.Module): The PyTorch model to compute forward pass for.
        criterion (nn.Module | None): Loss function. If None, loss is not computed.
        device (torch.device): Device on which to perform computations.

    Returns:
        Tuple[torch.Tensor | None, torch.Tensor]: Tuple containing computed loss
                                                  (None if criterion is None) and
                                                  model outputs.

    Raises:
        RuntimeError: If an error occurs while calculating the loss.
    """
    y_key = None

    # Check if inputs include target
    if isinstance(inputs, tuple) and len(inputs) == model.num_features + 1:
        # Unpack the input
        x_features, x_keys, y_key = inputs

        debug(f"Input features shape: {x_features.shape}")
        debug(f"Input keys shape: {x_keys.shape}")
        debug(f"Target batch shape: {y_key.shape}")

        # Move to device
        x_features = x_features.to(device)
        x_keys = x_keys.to(device)
        y_key = y_key.to(device)

        debug(f"Inputs moved to device: {device}")

        # Forward pass
        outputs = model(x_features, x_keys)

        debug(f"Model output shape: {outputs.shape}")
    else:
        # Assume inputs are directly
        # compatible with model
        outputs = model(*inputs)

        debug(f"Model output shape: {outputs.shape}")

    try:
        # Compute loss if criterion and target provided
        loss = None
        if criterion is not None and y_key is not None:
            # Calculate loss
            loss = criterion(outputs, y_key)

            debug(f"Loss computed: {loss.item()}")
            debug(f"Criterion used: {criterion}")
    except TypeError as e:
        msg = "Failed to compute loss at the end of forward pass"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Forward pass completed")

    return loss, outputs
