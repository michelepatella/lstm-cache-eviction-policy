from typing import Any, Dict

from tqdm import tqdm

from pipeline.config.pydantic.config import Config
from components.model.best.checks_updates.params_checker_updater import (
    check_update_best_model_params,
)
from components.validation.search_space.combinator import (
    get_parameters_combination,
)
from components.validation.time_series_cv.core.folds_runner import (
    compute_time_series_cv_folds,
)
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info


def compute_grid_search(
    training_set: AccessLogsDataset, config: Config
) -> Dict[str, Any]:
    """
    Perform grid search to find the
    best hyperparameter configuration.

    This function iterates over all parameter
    combinations generated from the search space, evaluates
    each combination using time-series cross-validation, and
    selects the one with the lowest average loss.

    Args:
        training_set (AccessLogsDataset): Training dataset to be used
                                          in CV evaluation.
        config (Config): Configuration object.

    Returns:
        Dict[str, Any]: Dictionary containing the best parameters found.
    """
    debug(f"Training set size for grid search: {len(training_set)}")

    # Prepare configuration
    cv_num_folds = config.validation.cross_validation.folds

    # Initialize best tracking variables
    best_params: Dict[str, Any] = {}
    best_avg_loss: float = float("inf")

    # Retrieve parameter combinations
    param_combinations = get_parameters_combination(config)

    # Iterate over all parameter combinations
    with tqdm(
        total=len(param_combinations),
        desc="Grid Search Progress",
    ) as pbar:
        for params in param_combinations:
            debug(f"Evaluating parameter combination: {params}")

            # Perform time series CV
            avg_loss = compute_time_series_cv_folds(
                cv_num_folds, training_set, params, config
            )

            debug(f"Average loss for current combination: {avg_loss}")

            # Update the best parameters
            # if improvement found
            best_avg_loss, best_params = check_update_best_model_params(
                avg_loss, best_avg_loss, params, best_params
            )

            pbar.update(1)

    info(f"Best parameters found: {best_params}")
    info(f"Best average loss found: {best_avg_loss}")

    info("Grid Search completed")

    return best_params
