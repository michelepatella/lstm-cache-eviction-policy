"""calculator.py

This module implements the custom loss calculation logic.

It combines a standard criterion loss with a temporal weighting mechanism
and a specific penalty term.

Functions:
    calculate_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> torch.Tensor | None:
        Computes the loss for a batch of predictions.
"""

import torch

from components.const import (
    EPSILON,
    LIST_FIRST_IDX,
    TENSOR_BROADCAST_COL_DIM,
    TENSOR_BROADCAST_ROW_DIM,
)
from components.logs.levels.error_logger import error


def calculate_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor | None:
    """Computes a custom loss.

    This function calculates a base loss using the provided criterion and
    modifies it by applying temporal weights derived from the distance to
    the next access of each key. It also adds a penalty term that
    specifically targets low-probability assignments for imminent accesses.

    Args:
        outputs (torch.Tensor): The model's raw output logits.
        targets (torch.Tensor): The ground truth of the accessed keys.
        criterion (torch.nn.Module): The base loss function.

    Returns:
        torch.Tensor | None: The computed final loss.

    Raises:
        RuntimeError: If loss calculation fails:
            * Criterion computation fails due to invalid input types or
              shapes (TypeError, RuntimeError).
    """
    try:
        if criterion is None:
            # None as loss if criterion is not provided
            final_loss = None
        else:
            # Retrieve batch size and device
            batch_size = len(targets)
            device = targets.device

            # Compute standard loss as base
            base_loss = criterion(outputs, targets)

            # Mask of future key accesses having true elements
            # where the same key appears later in the batch
            mask = targets.unsqueeze(
                TENSOR_BROADCAST_ROW_DIM,
            ) == targets.unsqueeze(TENSOR_BROADCAST_COL_DIM)

            # Initialize distances (to a large value)
            # which will store steps to next future access per key
            distances = torch.full(
                (batch_size,),
                batch_size + 1,
                device=device,
                dtype=torch.float,
            )

            # For each key, compute distances
            # to first future key access
            for i in range(batch_size):
                # Get indices of future accesses for key i
                future_accesses = torch.where(mask[i])[LIST_FIRST_IDX]
                future_accesses = future_accesses[future_accesses > i]

                # If there is a future access, store its
                # distance from current index
                if len(future_accesses) > 0:
                    distances[i] = (
                        future_accesses[LIST_FIRST_IDX] - i
                    ).float()

            # Calculate and normalize temporal weights
            # which assign higher values to keys that will
            # be accessed earlier than others
            temporal_weights = 1.0 / torch.log1p(distances)
            temporal_weights = temporal_weights / (
                temporal_weights.mean() + EPSILON
            )

            # Select logits of the target class for each batch
            # element and convert to probability via sigmoid
            target_probs = (
                torch.softmax(outputs, dim=TENSOR_BROADCAST_COL_DIM)
                .gather(
                    TENSOR_BROADCAST_COL_DIM,
                    targets.unsqueeze(TENSOR_BROADCAST_COL_DIM),
                )
                .squeeze(TENSOR_BROADCAST_COL_DIM)
                .clamp(min=EPSILON, max=1 - EPSILON)
            )

            # Penalize the model when assigning low
            # probabilities to keys that will be accessed
            # earlier than others
            penalty = (
                (1 - target_probs) * temporal_weights / temporal_weights.max()
            ).mean()

            # The final loss is given by weighting
            # the base one with temporal weights, adding a penalty
            final_loss = (base_loss * temporal_weights).mean() + penalty

        return final_loss
    except (TypeError, RuntimeError) as e:
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
