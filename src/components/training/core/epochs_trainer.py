import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.const import TRAINING_EPOCHS_DESC
from components.evaluation.model.evaluator import evaluate_model
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.best.checks_updates.checker_updater import (
    check_update_best_model,
)
from components.model.state_dict.copier import (
    copy_model_state_dict,
)
from components.training.callbacks.early_stopping import EarlyStopping
from components.training.core.single_epoch_trainer import train_single_epoch
from pipeline.config.pydantic.config import Config
from src.const import LOGS_VALIDATION_PHASE


def train_epochs(
    num_epochs: int,
    model: torch.nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    current_phase: str,
    config: Config,
) -> tuple[float, torch.nn.Module]:
    """Train a model for a given number of epochs.

    This function trains a given model for a specified number of epochs.
    The model is trained on the received training loader while validated
    with the validation one. Early stopping is applied to reduce computational
    costs and time. Training and evaluation process leverage optimizer and
    criterion received as arguments. All the operations are performed on a specified
    device.

    Args:
        num_epochs (int): Number of epochs to train.
        model (torch.nn.Module): The model to train.
        training_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer to use.
        criterion (torch.nn.Module): Loss function to use.
        device (torch.device): Device to run computations on.
        current_phase (str): Pipeline phase for which to run the training.
        config (Config): Configuration object.

    Returns:
        tuple[float, torch.nn.Module]:
            - best_avg_loss: The best average loss achieved.
            - model: The trained model with the best weights applied.

    Raises:
        RuntimeError: If training epochs fails:
            * Instantiation of EarlyStopping fails due to invalid input types
              (TypeError).
            * Iteration over epochs fails due to wrong number of epochs or
              invalid types (TypeError).
    """
    try:
        info(
            "Epochs training started",
            extra={
                "epochs_num": num_epochs,
                "training_loader_len": len(training_loader),
                "validation_loader_len": len(validation_loader),
                "optimizer": str(optimizer),
                "criterion": str(criterion),
                "device": str(device),
                "context": "Epochs training",
            },
        )

        # Prepare configuration
        num_features = config.model.general.features
        es_patience = (
            config.validation.early_stopping.patience
            if current_phase == LOGS_VALIDATION_PHASE
            else config.training.early_stopping.patience
        )
        es_delta = (
            config.validation.early_stopping.delta
            if current_phase == LOGS_VALIDATION_PHASE
            else config.training.early_stopping.delta
        )

        # Instantiate early stopping
        es = EarlyStopping(es_patience, es_delta)

        # Initialization
        best_model_weights = copy_model_state_dict(model)
        best_avg_loss = np.inf

        # Train the model over each epoch
        epoch = None
        for epoch in tqdm(range(1, num_epochs + 1), desc=TRAINING_EPOCHS_DESC):
            # Train one epoch
            train_single_epoch(
                model,
                training_loader,
                optimizer,
                criterion,
                device,
                epoch,
            )

            # Evaluate the model to get the
            # average loss after the current epoch
            avg_loss, *_ = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                num_features,
            )

            # Check for an update in average loss
            best_avg_loss, new_model_weights = check_update_best_model(
                avg_loss,
                best_avg_loss,
                model,
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
                break

        # Apply the best weights
        # before returning the trained model
        model.load_state_dict(best_model_weights)

        info(
            "Epochs training completed",
            extra={
                "loss_avg_best": None if np.isinf(best_avg_loss) or np.isnan(best_avg_loss) else float(best_avg_loss),
                "epochs_run_num": epoch,
                "early_stop_triggered": es.early_stop,
                "context": "Epochs training",
            },
        )

        return best_avg_loss, model
    except TypeError as e:
        msg = "Epochs training failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "epochs_num": num_epochs,
                "training_loader_len": len(training_loader),
                "validation_loader_len": len(validation_loader),
                "optimizer": str(optimizer),
                "criterion": str(criterion),
                "device": str(device),
                "context": "Epochs training",
            },
        )
        raise RuntimeError(msg) from e
