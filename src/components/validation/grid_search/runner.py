from typing import Dict, Tuple, Union

import mlflow
import numpy as np
from tqdm import tqdm

from components.const import GRID_SEARCH_DESC
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.best.checks_updates.params_checker_updater import (
    check_update_best_model_params,
)
from components.validation.search_space.combinator import (
    get_parameters_combination,
)
from components.validation.time_series_cv.core.folds_runner import (
    compute_time_series_cv_folds,
)
from pipeline.config.pydantic.config import Config
from src.const import LOGS_VALIDATION_PHASE, MLFLOW_NESTED_ENABLED


def compute_grid_search(
    training_set: AccessLogsDataset, config: Config
) -> Tuple[Dict[str, Union[int, float, bool]], float]:
    """
    Perform grid search to find the best parameters.

    This function iterates over all parameter combinations, evaluates
    each combination using time-series cross-validation, and selects
    the one achieving the lowest average loss.

    Args:
        training_set (AccessLogsDataset): Training dataset to be used in CV
                                          evaluation.
        config (Config): Configuration object.

    Returns:
        Tuple[Dict[str, Union[int, float, bool]], float]:
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
        cv_num_folds = config.validation.cross_validation.folds

        # Get all parameter combinations
        params_combinations = get_parameters_combination(config)

        info(
            "Grid search started",
            extra={
                "training_set_len": len(training_set),
                "total_param_combinations": len(params_combinations),
                "cv_num_folds": cv_num_folds,
                "context": "Grid search",
            },
        )

        # Initialize best tracking variables
        best_params = {}
        best_avg_loss = np.inf

        # Iterate over all parameter combinations
        with tqdm(
            total=len(params_combinations),
            desc=GRID_SEARCH_DESC,
        ) as pbar:
            for idx, params in enumerate(params_combinations, start=1):
                with mlflow.start_run(
                    run_name=f"{LOGS_VALIDATION_PHASE} ({idx}/{len(params_combinations)})",
                    nested=MLFLOW_NESTED_ENABLED,
                ):
                    # Perform time series CV
                    avg_loss, fold_losses = compute_time_series_cv_folds(
                        cv_num_folds, training_set, params, config
                    )

                    # Check and update the best parameters
                    # if improvement found
                    best_avg_loss, best_params = (
                        check_update_best_model_params(
                            avg_loss, best_avg_loss, params, best_params
                        )
                    )

                    # Experiment tracking
                    mlflow.log_params(params)
                    mlflow.log_metrics(
                        {
                            "loss_avg": avg_loss,
                            "loss_std": np.std(fold_losses),
                            "loss_min": np.min(fold_losses),
                            "loss_max": np.max(fold_losses),
                        }
                    )

                # To update the progress bar
                pbar.update(1)

        info(
            "Grid search completed",
            extra={
                "best_params": best_params,
                "best_avg_loss": best_avg_loss,
                "total_param_combinations": len(params_combinations),
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
                "num_folds": getattr(
                    config.validation.cross_validation, "folds", None
                ),
                "total_param_combinations": (
                    len(params_combinations)
                    if "params_combinations" in locals()
                    else None
                ),
                "current_best_avg_loss": (
                    best_avg_loss if "best_avg_loss" in locals() else None
                ),
                "context": "Grid search",
            },
        )
        raise RuntimeError(msg) from e
