from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from config.classes.Config import Config
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def get_last_accesses(
    current_idx: int, seq_len: int, testing_set: DataLoader, config: Config
) -> Tuple[Tensor, Tensor, Tensor] | None:
    """
    Extract the last sequence of accesses from the
    testing set.

    The function uses a sliding window of configured length
    ending at the current request index to extract the resulting
    last sequence of accesses from the testing set. If there is
    not enough data to form a full sequence, returns None.

    Args:
        current_idx (int): Current request index.
        seq_len (int): Sequence length to be extracted.
        testing_set (DataLoader): Testing dataset.
        config (Config): Configuration object.

    Returns:
        Tuple[Tensor, Tensor, Tensor] | None: A tuple containing the last
                                              sequence of accesses extracted,
                                              None if not enough data is available
                                              to extract last accesses of specified
                                              length.

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
        # into the sliding window defined
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

        # Convert selected testing
        # window to dataset
        window_dataset = AccessLogsDataset.from_dataframe(
            testing_window_df, config
        )

        # Get last accesses from
        # the testing window
        last_accesses = window_dataset[-1]

        debug(f"Last accesses extracted: {last_accesses}")

        info("Last accesses extracted")

        return last_accesses
    except (AttributeError, IndexError, KeyError, ValueError) as e:
        msg = "Failed to extract last accesses"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
