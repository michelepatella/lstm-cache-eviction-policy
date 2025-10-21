from typing import Optional

import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """
    Calculate the loss for outputs and targets.

    This function calculates the loss between model outputs
    and targets using the provided criterion.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        criterion (torch.nn.Module): Criterion to use for loss calculation.

    Returns:
        Optional[torch.Tensor]: Calculated loss (None if something went wrong).

    Raises:
        RuntimeError: If loss calculation fails:
            * Criterion computation fails due to invalid input types or
              shapes (TypeError, RuntimeError).
    """
    try:
        debug(f"Number of outputs for loss calculation: {len(outputs)}")
        debug(f"Number of targets for loss calculation: {len(targets)}")
        debug(f"Criterion for loss calculation: {criterion}")

        # Check whether criterion is not provided
        if criterion is None:
            # Loss is not calculated (None)
            loss = None
        else:
            # Calculate loss with
            # provided criterion
            loss = criterion(outputs, targets)

        debug(f"Loss calculated: {loss}")

        return loss
    except TypeError as e:
        msg = "Failed to calculate loss"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
