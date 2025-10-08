import copy
from typing import Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.classes.Config import Config
from pipeline.utils.evaluation.evaluator import evaluate_model
from pipeline.utils.training.callbacks.EarlyStopping import EarlyStopping
from pipeline.utils.training.one_epoch_trainer import train_one_epoch
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def train_n_epochs(
    num_epochs: int,
    model: torch.nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> Tuple[float, torch.nn.Module]:
    """
    Train a model for a given number of epochs.

    This function trains a given model for a specified
    number of epochs. The model is trained on the received
    training loader while validated with the validation one.
    Early stopping is applied to reduce computational costs and
    time. Training and evaluation process leverage optimizer and
    criterion received as arguments. All the operations are performed
    on a specified device.

    Args:
        num_epochs (int): Number of epochs to train.
        model (torch.nn.Module): The model to train.
        training_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer to use.
        criterion (nn.Module): Loss function to use.
        device (torch.device): Device to run training on.
        config (Config): Configuration object.

    Returns:
        Tuple[float, torch.nn.Module]: Tuple containing the best
                                       average loss achieved
                                       during training and the
                                       best trained model.

    Raises:
        RuntimeError: If an error occurs while deep copying the model weights.
    """
    debug(
        f"General training configuration:\n"
        f"- Number of epochs: {num_epochs}\n"
        f"- Training loader size: {len(training_loader)}\n"
        f"- Validation loader size: {len(validation_loader)}\n"
        f"- Optimizer: {optimizer}\n"
        f"- Criterion: {criterion}\n"
        f"- Device: {device}"
    )

    try:
        # Initialize model weights
        best_model_weights = copy.deepcopy(model.state_dict())
    except TypeError as e:
        msg = "Failed to deepcopy model weights"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Initialize bookkeeping
    best_avg_loss = float("inf")
    num_epochs_run = 0
    tot_loss = 0.0

    # Instantiate early stopping
    es = EarlyStopping(config)

    # Main training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        debug(f"Epoch started: {epoch}")

        # Train one epoch
        train_one_epoch(
            model, training_loader, optimizer, criterion, device, epoch
        )

        # Increase the number of epochs run
        num_epochs_run += 1

        # Evaluate the model to get the
        # average loss after the current epoch
        avg_loss, *_ = evaluate_model(
            model, validation_loader, criterion, device, config
        )

        debug(f"Validation average loss: {avg_loss:.6f}")

        # Add the current average loss
        # to the total one
        tot_loss += avg_loss

        # Save the best model weights if
        # an improvement is found
        if avg_loss < best_avg_loss:
            debug(
                f"New best average loss found ({avg_loss} < {best_avg_loss})"
            )

            # Both best average loss and best
            # model update are going to be updated
            # consequently
            best_avg_loss = avg_loss

            try:
                best_model_weights = copy.deepcopy(model.state_dict())
            except TypeError as e:
                msg = "Failed to deepcopy the current best model weights"
                error("%s: %s", msg, e)
                raise RuntimeError(msg) from e

        # Early stopping check
        # (check whether to stop training process
        # as the number of epochs without improvement
        # in average loss exceeds the patience)
        es(avg_loss)
        if es.early_stop:
            info("Early stopping triggered")
            break

    # Apply the best weights
    # before returning the trained model
    try:
        model.load_state_dict(best_model_weights)
    except TypeError as e:
        msg = "Failed to deepcopy the final best model weights"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Total number of epochs run: {num_epochs_run}")
    info(f"Best average loss achieved: {best_avg_loss}")
    info("Training process completed")

    return best_avg_loss, model
