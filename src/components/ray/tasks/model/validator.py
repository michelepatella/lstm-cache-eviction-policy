"""validator.py

Module containing the remote function for parallelized hyperparameter evaluation.

This module defines the `compute_time_series_cv_folds_task` function, designed
to be executed remotely using the Ray framework. It calculates the performance
of a single set of hyperparameters by executing Time Series Cross-Validation (TSCV)
across multiple folds of the training data. The results are returned to the main
process.

Functions:
    compute_time_series_cv_folds_task(
        training_set: AccessLogsDataset,
        params: dict[str, int | float | bool],
        config: Config
    ) -> tuple[dict[str, int | float | bool], float, list[float]]
        The remote function that performs TSCV for a given parameter set.
"""

import ray

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.validation.time_series_cv.core.folds_runner import (
    compute_time_series_cv_folds,
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
