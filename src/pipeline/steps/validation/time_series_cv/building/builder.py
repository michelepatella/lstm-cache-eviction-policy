from sklearn.model_selection import TimeSeriesSplit

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def build_time_series_split(num_folds: int) -> TimeSeriesSplit:
    """
    Build a TimeSeriesSplit instance.

    This function creates and returns a TimeSeriesSplit object
    configured with the specified number of folds.

    Args:
        num_folds (int): Number of folds (splits) to use for
                         time series.

    Returns:
        TimeSeriesSplit: Configured TimeSeriesSplit instance.

    Raises:
        RuntimeError: If an error occurs while building the
                      TimeSeriesSplit object, e.g.:
            * Invalid number of folds.
    """
    try:
        debug(f"Number of folders for TimeSeriesSplit: {num_folds}")

        # Instantiate the TimeSeriesSplit object
        tss = TimeSeriesSplit(n_splits=num_folds)

        info("TimeSeriesSplit built")

        return tss
    except ValueError as e:
        msg = "Failed to build TimeSeriesSplit"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
