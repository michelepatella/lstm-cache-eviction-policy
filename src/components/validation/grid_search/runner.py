from typing import Dict, Union

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


def compute_grid_search(
    training_set: AccessLogsDataset, config: Config
) -> Dict[str, Union[int, float, bool]]:
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
        Dict[str, Union[int, float, bool]]: Dictionary containing the best
                                            parameters found.

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
        debug(f"Training set size for grid search: {len(training_set)}")

        # Prepare configuration
        cv_num_folds = config.validation.cross_validation.folds

        # Get all parameter combinations
        params_combinations = get_parameters_combination(config)

        # Initialize best tracking variables
        best_params = {}
        best_avg_loss = np.inf

        # Iterate over all parameter combinations
        with tqdm(
            total=len(params_combinations),
            desc=GRID_SEARCH_DESC,
        ) as pbar:
            for params in params_combinations:
                debug(f"Evaluating parameters combination: {params}")

                # Perform time series CV
                avg_loss = compute_time_series_cv_folds(
                    cv_num_folds, training_set, params, config
                )

                # Check and update the best parameters
                # if improvement found
                best_avg_loss, best_params = check_update_best_model_params(
                    avg_loss, best_avg_loss, params, best_params
                )

                # To update the progress bar
                pbar.update(1)

        info(
            f"Grid search completed (best parameters: \n{best_params}\n"
            f"average loss: {best_avg_loss})"
        )

        return best_params
    except (
        TypeError,
        AttributeError,
        ValueError,
        FloatingPointError,
        RuntimeError,
    ) as e:
        msg = "Failed to compute grid search"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
