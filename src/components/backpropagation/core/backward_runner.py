import torch
from torch.optim import Optimizer

from components.logs.levels.info_logger import info


def compute_backward(loss: torch.nn.Module, optimizer: Optimizer) -> None:
    """
    Perform a backward pass and update model parameters.

    This function computes gradients via backpropagation
    from the provided loss and updates the model parameters
    using the given optimizer.

    Args:
        loss (torch.nn.Module): Loss tensor from which to compute gradients.
        optimizer (Optimizer): Optimizer used to update model parameters.

    Returns:
        None
    """
    # Compute gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    info("Backward pass completed")
