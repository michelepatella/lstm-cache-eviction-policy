from typing import Optional

import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def extract_sliding_window_dataset_rows(
    df: pd.DataFrame, idx: int, window_size: int
) -> Optional[pd.DataFrame]:
    """
    Extract a sliding window of rows from a dataset.

    This function extracts a sliding window of rows from a
    provided dataset, given a specific window size.

    Args:
        df (pd.DataFrame): The dataset to extract sliding windows from.
        idx (int): Index to center the window on.
        window_size (int): Number of rows to extract.

    Returns:
        Optional[pd.DataFrame]: The extracted window, or None if not enough rows.

    Raises:
        RuntimeError: If sliding window extraction from dataset fails:
            * Invalid arguments or data types (TypeError, ValueError).
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

        debug("Sliding window extracted from dataset")

        return window
    except (TypeError, ValueError) as e:
        msg = "Failed to extract dataset sliding window"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
