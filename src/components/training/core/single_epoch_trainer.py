import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.backpropagation.core.backward_runner import (
    compute_backward,
)
from components.backpropagation.core.forward_runner import compute_forward
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.mode.setter import set_model_mode
from const import TRAINING_MODEL_MODE, TRAINING_SINGLE_EPOCH_DESC


def train_single_epoch(
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
        RuntimeError: If training a single epoch fails:
            * One or more batches contain invalid shapes or unsupported
              data types (TypeError).
            * Forward pass fails during loss computation due to incompatible
              model output or criterion (TypeError).
            * Backward pass fails during gradient computation or parameter
              update (AttributeError or TypeError).
            * Optimizer fails to apply gradients due to internal
              incompatibility (AttributeError).
    """
    try:
        debug(
            f"Single epoch training configuration:\n"
            f"- Current epoch number: {epoch}\n"
            f"- Training loader size: {len(training_loader)}\n"
            f"- Optimizer: {optimizer}\n"
            f"- Criterion: {criterion}\n"
            f"- Device: {device}"
        )

        # To display the one-epoch progress
        training_loader_tqdm = tqdm(
            training_loader,
            desc=TRAINING_SINGLE_EPOCH_DESC + f"{epoch}",
        )

        # Set the model to training mode
        set_model_mode(model, TRAINING_MODEL_MODE)

        # For each batch in the training loader
        # run backpropagation algorithm
        for batch in training_loader_tqdm:
            # Reset the gradients
            optimizer.zero_grad()

            # Compute the forward pass to get
            # model outputs and calculate loss
            loss, _ = compute_forward(
                batch,
                model,
                device,
                criterion,
            )

            # Compute backward pass to update model
            # weights according to gradients of loss
            compute_backward(loss, optimizer)

            # Update progress bar and show
            # the current loss
            training_loader_tqdm.set_postfix(loss=loss)

        info("Training single epoch completed")
    except (TypeError, AttributeError) as e:
        msg = "Failed to perform training single epoch"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
