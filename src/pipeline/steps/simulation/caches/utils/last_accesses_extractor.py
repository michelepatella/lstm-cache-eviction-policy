from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from pipeline.const import (
    COS_TIME_COLUMN_NAME,
    REQUEST_COLUMN_NAME,
    SECONDS_IN_HOUR,
    SIN_TIME_COLUMN_NAME,
)
from pipeline.steps.simulation.utils.time_key_from_row_extractor import (
    extract_time_key_from_row,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def get_last_accesses(
    current_idx: int, seq_len: int, testing_set: DataLoader
) -> List[Tuple[float, int]] | None:
    """
    Extract the last sequence of accesses.

    This function extracts the last sequence of
    key accesses from the testing set, computing the access
    hour column from trigonometrically encoded time
    columns (cos time and sin time).

    Args:
        current_idx (int): Current request index.
        seq_len (int): Sequence length to extract.
        testing_set (DataLoader): Testing dataset.

    Returns:
        List[Tuple[float, int]]: Each tuple is (hour, key), None if no
                                 enough data is available.

    Raises:
        RuntimeError: If an error occurs during the extraction of
                      last accesses, e.g.:
                        * The testing set does not expose valid attributes.
                        * The computed sliding window indices are invalid.
                        * The dataset conversion from the testing window fails.
    """
    try:
        debug(
            f"Extracting last accesses at index "
            f"{current_idx} with sequence length {seq_len}"
        )

        # Sliding window indices
        start_idx = current_idx - seq_len + 1
        end_idx = current_idx + 1

        # Select data from testing set contained
        # into the sliding window
        testing_window_df = testing_set.data.iloc[start_idx:end_idx]

        debug(
            f"Testing window with start: {start_idx}, "
            f"end: {end_idx}, size: {len(testing_window_df)}"
        )

        # Check whether the testing window
        # select is less than the sequence length
        # to be extracted from it
        if len(testing_window_df) < seq_len:
            debug("Not enough testing data to extract last accesses")

            return None

        # Extract last accesses (hour and key) from
        # testing window
        last_accesses = []
        for _, row in testing_window_df.iterrows():
            # Extract tensors from the current row
            x_features = torch.tensor(
                row[[SIN_TIME_COLUMN_NAME, COS_TIME_COLUMN_NAME]].values,
                dtype=torch.float,
            )
            x_keys = torch.empty(0)
            y_key = torch.tensor(
                int(row[REQUEST_COLUMN_NAME]), dtype=torch.long
            )

            # Build the row tuple made up
            # of the extracted tensors
            row_tuple = (x_features, x_keys, y_key)

            # Extract time (in seconds) and key
            # from the current row
            current_time, key = extract_time_key_from_row(row_tuple)

            # Store the current access couple
            # (time in hours, key)
            last_accesses.append((current_time / SECONDS_IN_HOUR, key))

        debug(f"Extracted last accesses: {last_accesses}")
        info("Last accesses extracted successfully")

        return last_accesses
    except (AttributeError, IndexError, KeyError, ValueError) as e:
        msg = "Failed to extract last accesses"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
