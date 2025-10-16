from typing import Any
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def split_dataset_data(
    data: Any,
    split_idx: int,
    take_first: bool
) -> Any:
    """
    Split data at a given index.

    This function splits provided data based on
    the received index, taking the first or the second
    part of the data as specified.

    Args:
        data (Any): Data to split.
        split_idx (int): Index at which to split.
        take_first (bool): If True, take the first part of data.
                           If False, take the second part of data.

    Returns:
        Any: Split data.

    Raises:
        RuntimeError: If an error occurs while splitting dataset data e.g.:
            * Data does not support slicing or length calculations.
            * Split index is out of bounds.
    """
    debug(f"Split index to split dataset data: {split_idx}")
    debug(f"Take first for splitting dataset data: {take_first}")

    try:
        # Split data taking the first
        # or second part of it
        split_data = data[:split_idx] if take_first else data[split_idx:]

        info("Dataset data split")

        return split_data
    except (TypeError, IndexError, AttributeError) as e:
        msg = "Failed to split dataset data"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e