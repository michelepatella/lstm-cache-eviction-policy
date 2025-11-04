"""builder.py

Utility module for creating time series cross-validation splitters.

This module provides the `build_time_series_split` function, which
instantiates a `TimeSeriesSplit` object from scikit-learn, configured
with the specified number of folds. It is useful for evaluating models
on sequential data while preserving temporal order.

Functions:
    build_time_series_split(num_folds: int) -> TimeSeriesSplit
        Creates a TimeSeriesSplit object configured for the given number
        of folds for time-series cross-validation.
"""

from sklearn.model_selection import TimeSeriesSplit

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def build_time_series_split(num_folds: int) -> TimeSeriesSplit:
    """Build a TimeSeriesSplit object.

    This function creates and returns a TimeSeriesSplit object
    configured with the specified number of folds.

    Args:
        num_folds (int): Number of folds (splits) to use for time series.

    Returns:
        TimeSeriesSplit: Configured TimeSeriesSplit object.

    Raises:
        RuntimeError: If building the TimeSeriesSplit object fails:
            * Number of folds is invalid (ValueError).
            * Number of folds has incorrect type (TypeError).
    """
    try:
        debug(
            "TimeSeriesSplit building started",
            extra={
                "folds_requested_num": num_folds,
                "context": "TimeSeriesSplit building",
            },
        )

        # Instantiate the TimeSeriesSplit object
        tss = TimeSeriesSplit(n_splits=num_folds)

        debug(
            "TimeSeriesSplit building completed",
            extra={
                "folds_actual_num": tss.n_splits,
                "tss_object_type": type(tss).__name__,
                "context": "TimeSeriesSplit building",
            },
        )

        return tss
    except (ValueError, TypeError) as e:
        msg = "TimeSeriesSplit building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "folds_requested_num": num_folds,
                "context": "TimeSeriesSplit building",
            },
        )
        raise RuntimeError(msg) from e
