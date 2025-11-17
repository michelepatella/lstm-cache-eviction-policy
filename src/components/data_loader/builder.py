"""builder.py

Helper module for building PyTorch DataLoader instances.

This module provides the `build_data_loader` function, which wraps
torch.utils.data.DataLoader creation for a given dataset,
allowing configuration of batch size and shuffling.

Functions:
    build_data_loader(
        dataset: AccessLogsDataset | Dataset,
        batch_size: int,
        shuffle: bool = DATA_LOADER_SHUFFLE_DEFAULT,
        sampler: Sampler = None
    ) -> DataLoader
        Builds a DataLoader for the provided dataset with specified
        batch size and shuffle options.
"""

from torch.utils.data import DataLoader, Dataset, Sampler

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import DATA_LOADER_SHUFFLE_DEFAULT


def build_data_loader(
    dataset: AccessLogsDataset | Dataset,
    batch_size: int,
    shuffle: bool = DATA_LOADER_SHUFFLE_DEFAULT,
    sampler: Sampler = None,
) -> DataLoader:
    """Create a data loader for the given dataset.

    This function creates a data loader for a given dataset,
    applying specified settings including batch size and shuffling.

    Args:
        dataset (AccessLogsDataset | Dataset):
            The dataset instance to create the data loader for.
        batch_size (int): The batch size to use for the data loader.
        shuffle (bool): Whether to shuffle the data loader.
        sampler (Sampler): The sampler to use for the data loader.

    Returns:
        DataLoader: The data loader built.

    Raises:
        RuntimeError: If building the data loader fails:
            * Dataset is not compatible with torch.utils.data.Dataset
             (TypeError).
            * Batch size or shuffle parameters are invalid
              (TypeError, ValueError).
    """
    try:
        debug(
            "Data loader building started",
            extra={
                "dataset_type": type(dataset).__name__,
                "samples_num": (
                    len(dataset) if hasattr(dataset, "__len__") else None
                ),
                "batch_size": batch_size,
                "shuffle": shuffle,
                "context": "Data loader building",
            },
        )

        # Instantiate the data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )

        debug(
            "Data loader building completed",
            extra={
                "dataset_type": type(dataset).__name__,
                "samples_num": (
                    len(dataset) if hasattr(dataset, "__len__") else None
                ),
                "batch_size": batch_size,
                "shuffle": shuffle,
                "context": "Data loader building",
            },
        )

        return data_loader
    except (TypeError, ValueError) as e:
        msg = "Data loader building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dataset_type": type(dataset).__name__,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "context": "Data loader building",
            },
        )
        raise RuntimeError(msg) from e
