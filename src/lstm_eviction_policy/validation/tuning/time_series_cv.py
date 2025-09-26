from typing import Any, Dict, List

import numpy as np
from box import Box
from sklearn.model_selection import TimeSeriesSplit

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.data.dataloader.data_loader_builder import (
    create_data_loader,
)
from lstm_eviction_policy.utils.data.dataloader.dataloader_utils import (
    extract_targets_from_dataloader,
)
from lstm_eviction_policy.utils.data.dataset.dataset_splitter import (
    split_training_set,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, error, info
from lstm_eviction_policy.utils.model.setup.model_setup import model_setup
from lstm_eviction_policy.utils.training.n_epochs_trainer import train_n_epochs


def compute_time_series_cv(
    training_set: AccessLogsDataset,
    params: Dict[str, Any],
    config: Config,
) -> float:
    """
    Compute Time Series Cross-Validation (CV) average loss.

    This function splits the training set using time series cross-validation,
    trains a model for each fold, and calculates the average loss across folds.

    Parameters:
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
        f"Number of samples in training set for time series cross-validation: {num_samples}"
    )

    # Prepare configuration
    cv_num_folds = config.validation.cross_validation.folds
    training_shuffle = config.training.general.shuffle
    validation_shuffle = config.validation.general.shuffle

    debug(f"Number of folds for time series cross-validation: {cv_num_folds}")

    try:
        # Instantiate time series split object
        tscv = TimeSeriesSplit(n_splits=cv_num_folds)
        fold_losses: List[float] = []
    except ValueError as e:
        msg = "Failed to instantiate TimeSeriesSplit"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # For each fold
    for fold_idx, (train_idx, val_idx) in enumerate(
        tscv.split(np.arange(num_samples)), start=1
    ):
        try:
            # Box parameters
            params_box = Box(params)

            debug(f"Fold {fold_idx}, training index: {train_idx}")
            debug(f"Fold {fold_idx}, validation index: {val_idx}")

            # Split the whole training set into training
            # and validation sets
            training_dataset, validation_dataset = split_training_set(
                training_set,
                config,
                training_idx=train_idx,
                validation_idx=val_idx,
            )

            debug(f"Fold {fold_idx}, training size: {len(training_dataset)}")
            debug(
                f"Fold {fold_idx}, validation size: {len(validation_dataset)}"
            )

            # Create DataLoaders both for
            # training and validation sets
            training_loader = create_data_loader(
                training_dataset,
                config.training_batch_size,
                shuffle=training_shuffle,
            )
            validation_loader = create_data_loader(
                validation_dataset,
                config.training_batch_size,
                shuffle=validation_shuffle,
            )

            # Extract targets from dataset
            targets = extract_targets_from_dataloader(training_loader)

            # Setup model
            device, criterion, model, optimizer = model_setup(
                params_box.model,
                params_box.training.optimizer.learning_rate,
                targets,
                config,
            )

            # Train model
            avg_loss, _ = train_n_epochs(
                config.validation_num_epochs,
                model,
                training_loader,
                optimizer,
                criterion,
                device,
                config,
                validation_loader=validation_loader,
                early_stopping=True,
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
