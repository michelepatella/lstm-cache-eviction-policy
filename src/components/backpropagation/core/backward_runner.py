import torch
from torch.optim import Optimizer

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def compute_backward(loss: torch.Tensor, optimizer: Optimizer) -> None:
    """
    Perform a backward pass and update model parameters.

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
        debug(
            f"Computing backward pass with loss: "
            f"{loss} and optimizer: {optimizer}"
        )

        # Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        debug("Backward pass completed")
    except (RuntimeError, TypeError, ValueError) as e:
        msg = "Failed to compute backward pass"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
