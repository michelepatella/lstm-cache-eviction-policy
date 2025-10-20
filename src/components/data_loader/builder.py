from typing import Union

from torch.utils.data import DataLoader, Subset

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def build_data_loader(
    dataset: Union[AccessLogsDataset, Subset],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Create a data loader for the given dataset.

    This function creates a data loader for a given dataset (or subset of it),
    applying specified settings including batch size and shuffling.

    Args:
        dataset (Union[AccessLogsDataset, Subset]):
            The dataset instance to create the data loader for.
        batch_size (int): The batch size to use for the data loader.
        shuffle (bool): Whether to shuffle the data loader.

    Returns:
        DataLoader: The data loader built.

    Raises:
        RuntimeError: If building the data loader fails:
            * Dataset is not compatible with torch.utils.data.Dataset (TypeError).
            * Batch size or shuffle parameters are invalid (TypeError, ValueError).
    """
    try:
        debug(f"Batch size for the data loader to be built: {batch_size}")
        debug(f"Shuffle for the data loader to be built: {shuffle}")

        # Instantiate the data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        debug("Data loader built")

        return data_loader
    except (TypeError, ValueError) as e:
        msg = "Failed to build data loader"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
