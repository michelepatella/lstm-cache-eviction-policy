from torch.utils.data import DataLoader, Subset

from components.dataset.AccessLogsDataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def build_data_loader(
    dataset: AccessLogsDataset | Subset, batch_size: int, shuffle: bool
) -> DataLoader:
    """
    Create a data loader for the given dataset.

    This function creates a data loader for a given dataset,
    applying specified settings including batch size and shuffling.

    Args:
        dataset (AccessLogsDataset | Subset): The dataset instance
                                              to create the
                                              data loader for.
        batch_size (int): The batch size to use for the data loader.
        shuffle (bool): Whether to shuffle the data loader.

    Returns:
        DataLoader: The data loader built.

    Raises:
        RuntimeError: If an error occurs while creating
                      the data loader, e.g.:
            * If dataset object is not compatible
              with torch.utils.data.Dataset.
            * If batch size or shuffle parameters
              are of incorrect type or invalid.
    """
    debug(f"Batch size for the data loader to be built: {batch_size}")
    debug(f"Shuffle for the data loader to be built: {shuffle}")

    try:
        # Instantiate the data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    except (TypeError, ValueError) as e:
        msg = "Failed to create data loader"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Data loader built")

    return loader
