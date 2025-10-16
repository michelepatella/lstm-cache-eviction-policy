from typing import Dict, List

from pipeline.config.classes.Config import Config
from components.validation.time_series_cv.building.builder import (
    build_time_series_split,
)
from components.validation.time_series_cv.single_fold_runner import (
    compute_single_time_series_cv_fold,
)
from components.dataset.AccessLogsDataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average


def compute_time_series_cv_folds(
    cv_num_folds: int,
    training_set: AccessLogsDataset,
    params: Dict[str, int | float | bool],
    config: Config,
) -> float:
    """
    Compute Time Series Cross-Validation (CV) average loss.

    This function splits the training set using time series cross-validation,
    trains a model for each fold, and calculates the average loss across folds.

    Args:
        cv_num_folds (int): Number of folds for time series CV.
        training_set (AccessLogsDataset): Dataset to perform CV on.
        params (Dict[str, int | float | bool]): Hyperparameter configuration
                                                for model and training.
        config (Config): Configuration object.

    Returns:
        float: Average loss across all CV folds.
    """
    # Get the number of training set samples
    num_samples = len(training_set)

    debug(
        f"Training set size: {num_samples}, "
        f"Number of folds for time series CV: {cv_num_folds}"
    )

    # Build TimeSeriesSplit object
    tss = build_time_series_split(cv_num_folds)

    # Compute all fold indices
    fold_indices_list = list(tss.split(range(num_samples)))

    fold_losses: List[float] = []
    # For each fold
    for fold_idx, (train_idx, val_idx) in enumerate(
        fold_indices_list, start=1
    ):
        # Compute single fold
        avg_loss = compute_single_time_series_cv_fold(
            fold_idx, train_idx, val_idx, training_set, params, config
        )

        # Record fold average loss
        fold_losses.append(avg_loss)
        debug(f"Fold {fold_idx}, average loss: {avg_loss}")

    # Compute final average
    final_avg_loss = calculate_average(fold_losses)
    debug(f"Final average loss across all folds: {final_avg_loss}")

    info("Time series cross-validation completed")

    return final_avg_loss
