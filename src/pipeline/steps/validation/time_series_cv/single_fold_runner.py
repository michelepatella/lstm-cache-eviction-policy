from typing import Dict

import numpy as np
from box import Box

from pipeline.config.classes.Config import Config
from utils.data_loader.building.builder import build_data_loader
from utils.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.dataset.splitting.training_validation_splitter import (
    split_training_validation,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from utils.model.components.initializer import initialize_model_components
from utils.optimizer.builder import build_optimizer
from utils.training.epochs_trainer import train_epochs


def compute_single_time_series_cv_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    training_set: AccessLogsDataset,
    params: Dict[str, int | float | bool],
    config: Config,
) -> float:
    """
    Execute a single fold of time series cross-validation.

    This function handles all steps for one fold: dataset splitting,
    DataLoader creation, model initialization, optimizer building,
    model training, and average loss computation.

    Args:
        fold_idx (int): Index of the current fold.
        train_idx (np.ndarray): Indices for training samples in this fold.
        val_idx (np.ndarray): Indices for validation samples in this fold.
        training_set (AccessLogsDataset): Full training dataset.
        params (Dict[str, int | float | bool]): Hyperparameter configuration
                                                for this fold.
        config (Config): Configuration object.

    Returns:
        float: Average loss for the current fold.
    """
    # Box parameters for
    # easy attribute access
    params_box = Box(params)

    debug(f"Fold {fold_idx}, training index: {train_idx}")
    debug(f"Fold {fold_idx}, validation index: {val_idx}")

    # Split dataset for this fold into training
    # and validation sets
    training_dataset, validation_dataset = split_training_validation(
        training_set, training_idx=train_idx, validation_idx=val_idx
    )

    # Build DataLoaders for both sets
    training_loader = build_data_loader(
        training_dataset,
        config.training.general.batch_size,
        shuffle=config.training.general.shuffle,
    )
    validation_loader = build_data_loader(
        validation_dataset,
        config.validation.general.batch_size,
        shuffle=config.validation.general.shuffle,
    )

    # Extract targets for model initialization
    targets = extract_targets_from_data_loader(training_loader)

    # Initialize model components
    device, criterion, model = initialize_model_components(
        params_box.model,
        config,
        targets,
    )

    # Build optimizer
    optimizer = build_optimizer(
        model,
        config.training.optimizer.type,
        lr=config.training.optimizer.params.learning_rate,
        weight_decay=config.training.optimizer.params.weight_decay,
    )

    # Train model for the number of epochs in this fold
    avg_loss, _ = train_epochs(
        config.validation.cross_validation.epochs,
        model,
        training_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        config,
    )

    info(f"Single time series CV fold computation completed")

    return avg_loss
