import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def extract_dataset_sliding_window(
    df: pd.DataFrame, idx: int, window_size: int
) -> pd.DataFrame | None:
    """
    Extract a sliding window of rows from a dataset.

    This function extracts a sliding window of rows from a
    provided dataset, given a specific window size.

    Args:
        df (pd.DataFrame): The dataset to extract sliding windows from.
        idx (int): Index to center the window on.
        window_size (int): Number of rows to extract.

    Returns:
        pd.DataFrame | None: The extracted window, or None if not enough rows.

    Raises:
        RuntimeError: If the extraction fails due to invalid input
                      types or values, e.g.:
            * Provided dataset is not a pandas DataFrame.
            * Index or window size are not integers.
            * Window size is negative or zero.
    """
    try:
        # Determine the boundaries of the
        # sliding window, centered on
        # the provided index
        start_idx = idx - window_size + 1
        end_idx = idx + 1

        # Extract sliding window from dataset
        window = df.iloc[start_idx:end_idx]

        debug(
            f"Sliding window with start: {start_idx}, "
            f"end: {end_idx}, size: {len(window)}"
        )

        # Check whether the extracted window
        # size is less than requested
        if len(window) < window_size:
            debug("Sliding window size less than window size")
            return None

        info("Sliding window extracted from dataset")

        return window
    except (TypeError, ValueError) as e:
        msg = "Failed to extract dataset sliding window"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
