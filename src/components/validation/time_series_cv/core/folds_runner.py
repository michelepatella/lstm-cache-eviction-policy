from typing import Dict, List, Tuple, Union

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.math.avg_calculator import calculate_average
from components.validation.time_series_cv.builder import (
    build_time_series_split,
)
from components.validation.time_series_cv.core.single_fold_runner import (
    compute_single_time_series_cv_fold,
)
from pipeline.config.pydantic.config import Config


def compute_time_series_cv_folds(
    cv_num_folds: int,
    training_set: AccessLogsDataset,
    params: Dict[str, Union[int, float, bool]],
    config: Config,
) -> Tuple[float, List[float]]:
    """
    Compute Time Series Cross-Validation (CV).

    This function splits the training set using time series cross-validation,
    trains a model for each fold, and calculates the average loss across folds.

    Args:
        cv_num_folds (int): Number of folds for time series CV.
        training_set (AccessLogsDataset): Dataset to perform CV on.
        params (Dict[str, Union[int, float, bool]]): Parameter configuration
                                                     for model and training.
        config (Config): Configuration object.

    Returns:
        Tuple[float, List[float]]:
            - final_avg_loss: Final average loss over all fold losses.
            - fold_losses: List of all fold losses.

    Raises:
        RuntimeError: If time series cross-validation fails:
            * Training set is of wrong type or returns a non-integer
              length (TypeError).
            * The number of training set samples is not an integer
              (TypeError).
            * The calculated fold indices are not iterable (TypeError).
    """
    try:
        # Get the number of training set samples
        num_samples = len(training_set)

        # Build TimeSeriesSplit object
        tss = build_time_series_split(cv_num_folds)

        # Compute all fold indices
        fold_indices = list(tss.split(range(num_samples)))

        info(
            "Time Series CV started",
            extra={
                "cv_num_folds": cv_num_folds,
                "num_samples": num_samples,
                "num_fold_indices": len(fold_indices),
                "params": params,
                "training_set_type": type(training_set).__name__,
                "context": "Time Series CV",
            },
        )

        # Iterate over all folds
        fold_losses = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Compute single fold and get
            # its average loss
            avg_loss = compute_single_time_series_cv_fold(
                fold_idx, train_idx, val_idx, training_set, params, config
            )

            # Record current fold average loss
            fold_losses.append(avg_loss)

        # Compute final average loss for the current
        # parameters configuration
        final_avg_loss = calculate_average(fold_losses)

        info(
            "Time Series CV completed",
            extra={
                "final_avg_loss": final_avg_loss,
                "fold_losses": fold_losses,
                "num_folds_completed": len(fold_indices),
                "params": params,
                "context": "Time Series CV",
            },
        )

        return final_avg_loss, fold_losses
    except TypeError as e:
        msg = "Time Series CV failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "cv_num_folds": cv_num_folds,
                "training_set_type": type(training_set).__name__,
                "params": params,
                "context": "Time Series CV",
            },
        )
        raise RuntimeError(msg) from e
