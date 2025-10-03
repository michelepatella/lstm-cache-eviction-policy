from typing import Any, Dict

from tqdm import tqdm

from pipeline.config.classes.Config import Config
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info
from pipeline.validation.best_params.updater import (
    check_and_update_best_params,
)
from pipeline.validation.search_space.combinations.combinator import (
    get_parameters_combination,
)
from pipeline.validation.tuning.time_series_cv_runner import compute_time_series_cv


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

    Parameters:
        training_set (AccessLogsDataset): Training dataset to be used
                                          in CV evaluation.
        config (Config): Configuration object.

    Returns:
        Dict[str, Any]: Dictionary containing the best parameters found.
    """
    debug(f"Training set size for grid search: {len(training_set)}")

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
            avg_loss = compute_time_series_cv(training_set, params, config)

            debug(f"Average loss for current combination: {avg_loss}")

            # Update the best parameters
            # if improvement found
            best_avg_loss, best_params = check_and_update_best_params(
                avg_loss, best_avg_loss, params, best_params
            )

            pbar.update(1)

    info(f"Best parameters found: {best_params}")
    info(f"Best average loss found: {best_avg_loss}")

    info("Grid Search completed")

    return best_params
