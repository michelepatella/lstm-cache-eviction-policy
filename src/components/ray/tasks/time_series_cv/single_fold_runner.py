"""single_fold_runner.py

Module defining a Ray remote function for parallelized execution of a single
Time Series Cross-Validation (TSCV) fold.

This module provides the `compute_single_time_series_cv_fold_task` function,
designed to be executed in parallel via the Ray framework. Its purpose is
to efficiently isolate and execute the computationally intensive process of
training a model and evaluating its loss specifically on the data split defined
by a single train/validation fold.
Functions:
    compute_single_time_series_cv_fold_task(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        training_set: AccessLogsDataset,
        params: dict[str, int | float | bool],
        config: Config
    ) -> float
        The remote function that computes the loss for one specific TSCV fold.
"""

import numpy as np
import ray

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.validation.time_series_cv.core.single_fold_runner import (
    compute_single_time_series_cv_fold,
)
from pipeline.config.pydantic.config import Config


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
        train_idx,
        val_idx,
        training_set,
        params,
        config,
    )
    return avg_loss
