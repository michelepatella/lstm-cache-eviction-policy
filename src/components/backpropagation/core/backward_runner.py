import torch
from torch.optim import Optimizer

from components.logs.levels.error_logger import error


def compute_backward(loss: torch.Tensor, optimizer: Optimizer) -> None:
    """Perform a backward pass and update model parameters.

    This function computes gradients via backpropagation from the
    provided loss and updates the model parameters using the given optimizer.

    Args:
        loss (torch.Tensor): Loss tensor from which to compute gradients.
        optimizer (Optimizer): Optimizer used to update model parameters.

    Returns:
        None

    Raises:
    RuntimeError: If backward pass fails:
        * Gradient computation fails due to invalid loss tensor or autograd
          errors (RuntimeError).
        * Optimizer update fails because optimizer received invalid types or
          parameters (TypeError, ValueError).
    """
    try:
        # Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()
    except (RuntimeError, TypeError, ValueError) as e:
        msg = "Backward pass failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "loss_type": type(loss).__name__,
                "loss_shape": (
                    tuple(loss.shape)
                    if isinstance(loss, torch.Tensor)
                    else None
                ),
                "loss_requires_grad": getattr(loss, "requires_grad", None),
                "optimizer_type": type(optimizer).__name__,
                "optimizer_param_groups": getattr(
                    optimizer,
                    "param_groups",
                    None,
                ),
                "context": "Model backward pass",
            },
        )
        raise RuntimeError(msg) from e
