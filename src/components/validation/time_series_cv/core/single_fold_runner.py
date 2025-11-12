"""single_fold_runner.py

Core utility module for executing a single fold of time series
cross-validation.

This module provides the `compute_single_time_series_cv_fold` function,
which orchestrates all steps for one CV fold: dataset splitting,
DataLoader creation, model environment setup, optimizer building,
and model training, returning the resulting average loss.

Functions:
    compute_single_time_series_cv_fold(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        training_set: AccessLogsDataset,
        params: dict[str, int | float | bool],
        pipeline_config: PipelineConfig
    ) -> float
        Executes a single fold of time series cross-validation and
        returns the average loss for that fold.
"""

import numpy as np

from components.data_loader.builder import build_data_loader
from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from components.logs.initializer import logs_phase
from components.model.environment.initializer import (
    initialize_model_environment,
)
from components.optimizer.builder import build_optimizer
from components.training.core.epochs_trainer import train_epochs
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def compute_single_time_series_cv_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    training_set: AccessLogsDataset,
    params: dict[str, int | float | bool],
    pipeline_config: PipelineConfig,
) -> float:
    """Execute a single fold of time series cross-validation.

    This function handles all steps for time series cross-validation on
    one fold: dataset splitting, DataLoader creation, model environment
    initialization, optimizer building, and model training with resulting
    average loss.

    Args:
        train_idx (np.ndarray): Indices for training samples in this fold.
        val_idx (np.ndarray): Indices for validation samples in this fold.
        training_set (AccessLogsDataset): Full training set.
        params (dict[str, int | float | bool]): Parameters configuration
                                                     for the current fold.
        pipeline_config (PipelineConfig): Configuration object.

    Returns:
        float: Average loss for the current fold.
    """
    # Prepare configuration
    training_batch_size = pipeline_config.data_loader.batch_size.training
    training_shuffle = pipeline_config.data_loader.shuffle.training
    validation_batch_size = pipeline_config.data_loader.batch_size.validation
    validation_shuffle = pipeline_config.data_loader.shuffle.validation
    validation_device = pipeline_config.resources.devices.validation
    optimizer_type = pipeline_config.optimizer.type
    learning_rate = pipeline_config.optimizer.params.learning_rate
    weight_decay = pipeline_config.optimizer.params.weight_decay
    num_epochs = pipeline_config.validation.time_series_cv.epochs

    # Split training set into training and validation sets
    training_set, validation_set = split_training_validation_sets(
        training_set,
        training_idx=train_idx,
        validation_idx=val_idx,
    )

    # Build DataLoaders for both sets
    training_loader = build_data_loader(
        training_set,
        training_batch_size,
        training_shuffle,
    )
    validation_loader = build_data_loader(
        validation_set,
        validation_batch_size,
        validation_shuffle,
    )

    # Extract targets for model initialization
    targets = extract_targets_from_data_loader(training_loader)

    # Initialize model environment
    device, criterion, model = initialize_model_environment(
        targets,
        validation_device,
        pipeline_config,
        model_params=params,
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
        num_epochs,
        model,
        training_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        logs_phase.get(),
        pipeline_config,
    )

    return avg_loss
