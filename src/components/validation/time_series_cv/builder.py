from sklearn.model_selection import TimeSeriesSplit

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def build_time_series_split(num_folds: int) -> TimeSeriesSplit:
    """
    Build a TimeSeriesSplit object.

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
        debug(f"Number of folds for TimeSeriesSplit: {num_folds}")

        # Instantiate the TimeSeriesSplit object
        tss = TimeSeriesSplit(n_splits=num_folds)

        debug(f"TimeSeriesSplit built")

        return tss
    except (ValueError, TypeError) as e:
        msg = "Failed to build TimeSeriesSplit"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
