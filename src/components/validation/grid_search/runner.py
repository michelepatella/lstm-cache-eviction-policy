"""runner.py

Module for performing grid search over model hyperparameters.

This module provides the hyperparameter tuning of the model. It leverages
the Ray framework to parallelize the evaluation of numerous hyperparameter
combinations. The module aims at assessing the parameters combination's
performance using Time Series Cross-Validation (TSCV) against a training dataset.
It identifies and returns the set of parameters that yield the lowest average TSCV loss.

Functions:
    compute_grid_search(
        training_set: AccessLogsDataset,
        params_combinations: list[dict[str, int | float | bool]],
        config: Config
    ) -> tuple[dict[str, int | float | bool], float]
        Executes the parallelized grid search, evaluates parameter combinations
        using TSCV, and determines the optimal hyperparameter set.
"""

import mlflow
import numpy as np
import ray

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.best.checks_updates.params_checker_updater import (
    check_update_best_model_params,
)
from components.ray.tasks.time_series_cv.folds_runner import (
    compute_time_series_cv_folds_task,
)
from const import LOGS_PHASE_VALIDATION, MLFLOW_NESTED
from pipeline.config.pydantic.config import Config


def compute_grid_search(
    training_set: AccessLogsDataset,
    params_combinations: list[dict[str, int | float | bool]],
    config: Config,
) -> tuple[dict[str, int | float | bool], float]:
    """Perform grid search to find the best parameters.

    This function iterates over all parameter combinations, evaluates
    each combination using time-series cross-validation, and selects
    the one achieving the lowest average loss.

    Args:
        training_set (AccessLogsDataset): Training dataset to be used in CV
                                          evaluation.
        params_combinations (list[dict[str, int | float | bool]]):
            Parameter combinations to be evaluated.
        config (Config): Configuration object.

    Returns:
        tuple[dict[str, int | float | bool], float]:
            - best_params: Dictionary containing the best parameters found.
            - best_avg_loss: Average loss of the best parameters.

    Raises:
        RuntimeError: If grid search computation fails:
            * Parameter combinations cannot be generated due to invalid
              configuration fields or types (TypeError or AttributeError).
            * Time-series cross-validation fails due to inconsistent data,
              invalid fold splitting, or incompatible model parameters
              (ValueError or TypeError).
            * Loss computation during CV fails because of unexpected output
              values or numerical instability (FloatingPointError or ValueError).
            * Best parameters update fails because of incompatible
              comparison between losses or invalid parameter dictionary
              structure (TypeError).
            * Progress bar update or iteration fails due to internal tqdm
              error or invalid iterable (RuntimeError or TypeError).
    """
    try:
        # Prepare configuration
        cv_num_folds = config.validation.time_series_cv.folds

        info(
            "Grid search started",
            extra={
                "training_set_len": len(training_set),
                "param_combinations_num": len(params_combinations),
                "folds_num": cv_num_folds,
                "context": "Grid search",
            },
        )

        # Initialize best tracking variables
        best_params = {}
        best_avg_loss = np.inf

        # Parallelize params tuning through Ray
        # and get the final results coming from remote
        futures = [
            compute_time_series_cv_folds_task.remote(
                training_set,
                params,
                config,
            )
            for params in params_combinations
        ]
        results = ray.get(futures)

        for idx, (params, avg_loss, fold_losses) in enumerate(
            results,
            start=1,
        ):
            with mlflow.start_run(
                run_name=f"{LOGS_PHASE_VALIDATION} ({idx})",
                nested=MLFLOW_NESTED,
            ):
                # Check and update the best parameters
                # if improvement found
                best_avg_loss, best_params = check_update_best_model_params(
                    avg_loss,
                    best_avg_loss,
                    params,
                    best_params,
                )

                # Experiment tracking
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "loss_avg": float(avg_loss),
                        "loss_std": np.std(fold_losses),
                        "loss_min": np.min(fold_losses),
                        "loss_max": np.max(fold_losses),
                    },
                )

        info(
            "Grid search completed",
            extra={
                "params_best": best_params,
                "loss_avg_best": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "param_combinations_num": len(params_combinations),
                "context": "Grid search",
            },
        )

        return best_params, best_avg_loss
    except (
        TypeError,
        AttributeError,
        ValueError,
        FloatingPointError,
        RuntimeError,
    ) as e:
        msg = "Grid search failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "training_set_len": (
                    len(training_set) if training_set else None
                ),
                "folds_num": getattr(
                    config.validation.time_series_cv.folds,
                    "folds",
                    None,
                ),
                "param_combinations_num": (
                    len(params_combinations)
                    if "params_combinations" in locals()
                    else None
                ),
                "context": "Grid search",
            },
        )
        raise RuntimeError(msg) from e
