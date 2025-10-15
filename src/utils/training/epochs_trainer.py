from typing import Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.config.classes.Config import Config
from utils.evaluation.evaluator import evaluate_model
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from utils.model.best.updater import (
    update_best_model,
)
from utils.model.state_dict.copier import (
    copy_model_state_dict,
)
from utils.training.callbacks.EarlyStopping import EarlyStopping
from utils.training.single_epoch_trainer import train_single_epoch


def train_epochs(
    num_epochs: int,
    model: torch.nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
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
        criterion (torch.nn.Module): Loss function to use.
        device (torch.device): Device to run training on.
        config (Config): Configuration object.

    Returns:
        Tuple[float, torch.nn.Module]: Tuple containing the best
                                       average loss achieved
                                       during training and the
                                       best trained model.
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

    # Initialization
    best_model_weights = copy_model_state_dict(model)
    best_avg_loss = float("inf")

    # Prepare configuration
    num_features = config.model.general.features

    # Instantiate early stopping
    es = EarlyStopping(config)

    # Main training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        debug(f"Epoch started: {epoch}")

        # Train one epoch
        train_single_epoch(
            model, training_loader, optimizer, criterion, device, epoch
        )

        # Evaluate the model to get the
        # average loss after the current epoch
        avg_loss, *_ = evaluate_model(
            model, validation_loader, criterion, device, num_features
        )

        debug(f"Validation average loss: {avg_loss}")

        # Check for an update in average loss
        best_avg_loss, new_model_weights = update_best_model(
            avg_loss, best_avg_loss, model
        )

        # Update best model weights (if any)
        if new_model_weights:
            best_model_weights = new_model_weights

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
    model.load_state_dict(best_model_weights)

    info(f"Best average loss achieved: {best_avg_loss}")
    info("Training process completed")

    return best_avg_loss, model
