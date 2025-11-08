"""validator.py

Module defining Ray remote functions for parallelized Time Series Cross-Validation
(TSCV).

This module provides remote tasks for hyperparameter evaluation, by leveraging the
Ray framework to distribute the computationally intensive operations.

Functions:
    compute_time_series_cv_folds_task(
        training_set: AccessLogsDataset,
        params: dict[str, int | float | bool],
        config: Config
    ) -> tuple[dict[str, int | float | bool], float, list[float]]
        The remote function that performs TSCV across all folds for a given parameter set.
    compute_single_time_series_cv_fold_task(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        training_set: AccessLogsDataset,
        params: dict[str, int | float | bool],
        config: Config
    ) -> float
        The remote function that computes the loss for a single TSCV fold.
"""

import numpy as np
import ray

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.validation.time_series_cv.core.folds_runner import (
    compute_time_series_cv_folds,
)
from components.validation.time_series_cv.core.single_fold_runner import (
    compute_single_time_series_cv_fold,
)
from pipeline.config.pydantic.config import Config


@ray.remote
def compute_time_series_cv_folds_task(
    training_set: AccessLogsDataset,
    params: dict[str, int | float | bool],
    config: Config,
) -> tuple[dict[str, int | float | bool], float, list[float]]:
    """Computes the Time Series Cross-Validation for a given set
    of parameters.

    This function is executed remotely via Ray. It calculates the loss by applying
    Time Series Cross-Validation across multiple folds on the training data.

    Args:
        training_set (AccessLogsDataset): The dataset containing access logs
                                          used for training and validation.
        params (dict[str, int | float | bool]): The model parameters to be
                                                evaluated in this run.
        config (Config): The configuration object.

    Returns:
        tuple[dict[str, int | float | bool], float]:
            - params: The model parameters evaluated.
            - avg_loss: The average loss of the current model parameters.
            - fold_loss: The average loss of all the folds.
    """
    # Compute a Time Series CV over the
    # given model parameters
    avg_loss, fold_losses = compute_time_series_cv_folds(
        config.validation.time_series_cv.folds, training_set, params, config
    )

    return params, avg_loss, fold_losses


@ray.remote
def compute_single_time_series_cv_fold_task(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    training_set: AccessLogsDataset,
    params: dict[str, int | float | bool],
    config: Config,
) -> float:
    """Computes the loss for a single fold within the Time Series Cross-Validation.

    This function is executed remotely via Ray. It uses the provided training
    and validation indices to split the dataset, trains the model with the given
    parameters on the training subset, and evaluates the loss on the validation subset.

    Args:
        train_idx (np.ndarray): Array of indices defining the training samples
                                for this fold.
        val_idx (np.ndarray): Array of indices defining the validation samples
                              for this fold.
        training_set (AccessLogsDataset): The full dataset used for splitting.
        params (dict[str, int | float | bool]): The model parameters to be used
                                                for training.
        config (Config): The configuration object.

    Returns:
        float: The computed loss value for the specific validation fold.
    """
    avg_loss = compute_single_time_series_cv_fold(
        train_idx, val_idx, training_set, params, config
    )
    return avg_loss
