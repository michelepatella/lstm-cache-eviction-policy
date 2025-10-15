import torch

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def calculate_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor | None:
    """
    Calculate the loss.

    This function calculates the loss between model outputs
    and target tensors using the provided criterion.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth target tensor.
        criterion (torch.nn.Module): Criterion to use for loss calculation.

    Returns:
        torch.Tensor | None: Calculated loss (None if something went wrong).

    Raises:
        RuntimeError: If loss calculation fails e.g.:
            * Invalid input types.
    """
    try:
        # Check whether criterion is not provided
        if criterion is None:
            # Loss is not calculated (None)
            loss = None
        else:
            # Calculate loss
            loss = criterion(outputs, targets)

        debug(f"Loss calculated: {loss.item()}, using criterion: {criterion}")
        info("Loss calculation completed")

        return loss
    except TypeError as e:
        msg = "Failed to calculate loss"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
