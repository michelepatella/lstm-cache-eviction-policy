import torch
from torch.utils.data import DataLoader

from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.error_logger import error
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def extract_targets_from_data_loader(data_loader: DataLoader) -> torch.Tensor:
    """
    Extract all target tensors from a data
    loader and concatenate them.

    This function iterates through a DataLoader and collects
    all target tensors. The collected targets are then concatenated
    into a single tensor.

    Parameters:
        data_loader (DataLoader): DataLoader instance from which
                                  to extract targets.

    Returns:
        torch.Tensor: Concatenated tensor of all targets.

    Raises:
        RuntimeError: If an error occurs during target extraction, e.g.:
            * Dataset yields unexpected batch structure.
            * Any error during tensor concatenation.
    """
    try:
        # Initialization
        all_targets = []

        # Extract all targets from data loader
        for batch in data_loader:
            targets = batch[-1]
            all_targets.append(targets)

        # Concatenate extracted targets
        concatenated_targets = torch.cat(all_targets)

        debug(f"Concatenated targets shape: {concatenated_targets.shape}")
    except (TypeError, ValueError, IndexError) as e:
        msg = "Failed to extract targets from data loader"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Target extraction from data loader completed")

    return concatenated_targets
