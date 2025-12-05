"""calculator.py

Loss calculation utility module.

This module provides the `calculate_loss` function, which computes
the loss between model outputs and targets using a provided PyTorch
criterion. The function also handles cases where the criterion may be
None.

Functions:
    calculate_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: torch.nn.Module
    ) -> torch.Tensor | None
        Calculates the loss between model outputs and targets. Returns
        None if the criterion is not provided.
"""

import torch

from components.logs.levels.error_logger import error


def calculate_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor | None:
    """Calculate the loss for outputs and targets.

    This function calculates the loss between model outputs
    and targets using the provided criterion.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        criterion (torch.nn.Module): Criterion to use for loss calculation.

    Returns:
        torch.Tensor | None: Calculated loss (None if something went wrong).

    Raises:
        RuntimeError: If loss calculation fails:
            * Criterion computation fails due to invalid input types or
              shapes (TypeError, RuntimeError).
    """
    try:
        # Check whether criterion is not provided
        if criterion is None:
            # Loss is not calculated (None)
            loss = None
        else:
            # Calculate loss with
            # provided criterion
            loss = criterion(outputs, targets)

        return loss
    except TypeError as e:
        msg = "Loss calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "outputs_type": type(outputs).__name__,
                "outputs_shape": (
                    tuple(outputs.shape)
                    if isinstance(outputs, torch.Tensor)
                    else None
                ),
                "targets_type": type(targets).__name__,
                "targets_shape": (
                    tuple(targets.shape)
                    if isinstance(targets, torch.Tensor)
                    else None
                ),
                "criterion_type": (
                    type(criterion).__name__ if criterion is not None else None
                ),
                "context": "Loss calculation",
            },
        )
        raise RuntimeError(msg) from e
