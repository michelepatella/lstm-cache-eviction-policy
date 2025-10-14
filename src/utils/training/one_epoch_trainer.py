import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.backpropagation.backward_runner import (
    compute_backward,
)
from utils.backpropagation.forward_runner import compute_forward
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def train_one_epoch(
    model: torch.nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> None:
    """
    Train the model for a single epoch.

    This function performs one full pass over the training set.
    For each batch in the training set this function performs both
    a forward pass — to obtain predictions and loss — as well
    as a backward pass — to calculate the gradients of loss and update
    the weights through optimizer consequently.

    Args:
        model (torch.nn.Module): The model to train.
        training_loader (DataLoader): DataLoader providing training batches.
        optimizer (Optimizer): Optimizer used to update model parameters.
        criterion (torch.nn.Module): Loss function used to compute the loss.
        device (torch.device): Device on which training is performed.
        epoch (int): Current epoch number for logging and progress display.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs during processing of a batch e.g.:
            * Invalid batch shapes.
            * Type errors.
            * Loss computation failure.
    """
    debug(
        f"One-epoch training configuration:\n"
        f"- Current epoch number: {epoch}\n"
        f"- Training loader size: {len(training_loader)}\n"
        f"- Optimizer: {optimizer}\n"
        f"- Criterion: {criterion}\n"
        f"- Device: {device}"
    )

    # To display the one-epoch progress
    training_loader = tqdm(
        training_loader,
        desc=f"Epoch {epoch}",
        leave=False,
    )

    # Set the model to training mode
    model.train()

    # For each batch in the training loader
    # run backpropagation algorithm
    for batch in training_loader:
        try:
            debug(
                f"Current batch shapes during "
                f"one-epoch training: {[t.shape for t in batch]}"
            )

            # Reset the gradients
            optimizer.zero_grad()

            # Compute the forward pass
            # (to calculate the output — moving from
            # input layer to output layer — and the loss
            # after comparing model prediction with expected one)
            loss, _ = compute_forward(
                batch,
                model,
                criterion,
                device,
            )

            debug(
                f"Current batch loss during one-epoch training: {loss.item()}"
            )

            # Compute backward pass
            # (to calculate the gradients of loss with
            # respect to the weights and update weights
            # consequently)
            compute_backward(loss, optimizer)

            # Update progress bar and show
            # the current loss
            training_loader.set_postfix(loss=loss.item())
        except (TypeError, AttributeError, ValueError) as e:
            msg = "Failed to compute one-epoch training"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    info("One-epoch training completed")
