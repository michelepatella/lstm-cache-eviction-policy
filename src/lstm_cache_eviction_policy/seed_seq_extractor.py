from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from config.classes.Config import Config
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def extract_seed_seq(
    current_idx: int, testing_set: DataLoader, config: Config
) -> Tuple[Tensor, Tensor, Tensor] | None:
    """
    Extract the seed sequence used by the autoregressive rollout.

    This function extracts a seed sequence from the testing dataset
    based on a sliding window determined by the current index and
    the configured sequence length.

    Parameters:
        current_idx (int): Current request index in the dataset.
        testing_set: Testing dataset object containing access log data.
        config (Config): Configuration object containing sequence parameters.

    Returns:
        Any: Seed sequence extracted from the testing window.

    Raises:
        RuntimeError: If an unexpected error occurs during extraction.
    """
    try:
        debug(
            f"Seed sequence extraction at index: {current_idx}, "
            f"sequence length: {config.data.sequence.length}"
        )

        # Compute sliding window boundaries
        start_idx = current_idx - config.data.sequence.length + 1
        end_idx = current_idx + 1

        # Select window from testing dataset
        testing_window_df = testing_set.data.iloc[start_idx : end_idx + 1]
        debug(
            f"Testing window indices: start={start_idx}, end={end_idx}, "
            f"size={len(testing_window_df)}"
        )

        # Ensure enough data for sequence extraction
        if len(testing_window_df) < config.data.sequence.length:
            info("Not enough data to extract seed sequence — skipping.")
            return None

        # Build dataset from testing window
        testing_window_dataset = AccessLogsDataset.from_dataframe(
            testing_window_df, config
        )
        debug(f"Testing window dataset size: {len(testing_window_dataset)}")

        # Ensure dataset is not empty
        if len(testing_window_dataset) == 0:
            info("Testing window dataset is empty — skipping.")
            return None

        # Extract last element as seed sequence
        seed_seq = testing_window_dataset[len(testing_window_dataset) - 1]
        info("Seed sequence successfully extracted.")

        return seed_seq

    except Exception as e:
        msg = "Failed to extract seed sequence from testing set"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
