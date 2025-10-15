from typing import Any, Dict, List

import numpy as np
from box import Box

from pipeline.config.classes.Config import Config
from pipeline.steps.validation.time_series_cv.builder import build_time_series_split
from utils.data_loader.building.builder import (
    build_data_loader,
)
from utils.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.dataset.building.training_validation_splitter import (
    split_training_validation,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.model.components.initializer import initialize_model_components
from utils.optimizer.builder import build_optimizer
from utils.training.epochs_trainer import train_epochs


def compute_time_series_cv(
    training_set: AccessLogsDataset,
    params: Dict[str, Any],
    config: Config,
) -> float:
    """
    Compute Time Series Cross-Validation (CV) average loss.

    This function splits the training set using time series cross-validation,
    trains a model for each fold, and calculates the average loss across folds.

    Args:
        training_set (AccessLogsDataset): Dataset to perform CV on.
        params (Dict[str, Any]): Hyperparameter configuration
                                 for model and training.
        config (Config): Configuration object.

    Returns:
        float: Average loss across all CV folds.

    Raises:
        RuntimeError: If an error occurs during CV computation e.g.:
            * TimeSeriesSplit creation.
            * Dataset splitting.
            * DataLoader creation.
            * Model setup.
            * Model training.
            * Final average loss computation.
    """
    # Get the number of training set samples
    num_samples = len(training_set)

    debug(
        f"Number of samples in training set "
        f"for time series cross-validation: {num_samples}"
    )

    # Prepare configuration
    cv_num_folds = config.validation.cross_validation.folds
    cv_num_epochs = config.validation.cross_validation.epochs
    training_shuffle = config.training.general.shuffle
    validation_shuffle = config.validation.general.shuffle
    optimizer_type = config.training.optimizer.type
    learning_rate = config.training.optimizer.params.learning_rate
    weight_decay = config.training.optimizer.params.weight_decay

    debug(f"Number of folds for time series cross-validation: {cv_num_folds}")

    # Build TimeSeriesSplit object
    tss = build_time_series_split(cv_num_folds)

    fold_losses: List[float] = []
    # For each fold
    for fold_idx, (train_idx, val_idx) in enumerate(
        tss.split(np.arange(num_samples)), start=1
    ):
        try:
            # Box parameters
            params_box = Box(params)

            debug(f"Fold {fold_idx}, training index: {train_idx}")
            debug(f"Fold {fold_idx}, validation index: {val_idx}")

            # Split the whole training set into training
            # and validation sets
            training_dataset, validation_dataset = split_training_validation(
                training_set,
                training_idx=train_idx,
                validation_idx=val_idx,
            )

            debug(f"Fold {fold_idx}, training size: {len(training_dataset)}")
            debug(
                f"Fold {fold_idx}, validation size: {len(validation_dataset)}"
            )

            # Create DataLoaders both for
            # training and validation sets
            training_loader = build_data_loader(
                training_dataset,
                config.training.general.batch_size,
                shuffle=training_shuffle,
            )
            validation_loader = build_data_loader(
                validation_dataset,
                config.validation.general.batch_size,
                shuffle=validation_shuffle,
            )

            # Extract targets from dataset
            targets = extract_targets_from_data_loader(training_loader)

            # Setup model
            device, criterion, model = initialize_model_components(
                params_box.model,
                config,
                targets,
            )

            # Build optimizer
            optimizer = build_optimizer(
                model,
                optimizer_type,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # Train model
            avg_loss, _ = train_epochs(
                cv_num_epochs,
                model,
                training_loader,
                validation_loader,
                optimizer,
                criterion,
                device,
                config,
            )

            # Record fold average loss
            if avg_loss is not None:
                fold_losses.append(avg_loss)
                debug(f"Fold {fold_idx}, average loss: {avg_loss}")
        except (IndexError, ValueError, TypeError, RuntimeError) as e:
            msg = "Failed to compute time series cross-validation"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    # Compute final average
    try:
        final_avg_loss = float(np.mean(fold_losses))
        debug(f"Final average loss across all folds: {final_avg_loss}")
    except (TypeError, ValueError) as e:
        msg = "Failed to compute final average loss across all folds"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Time series cross-validation completed")

    return final_avg_loss
