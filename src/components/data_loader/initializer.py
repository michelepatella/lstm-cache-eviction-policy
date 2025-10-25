from typing import Tuple

from torch.utils.data import DataLoader

from components.data_loader.builder import build_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from pipeline.config.pydantic.config import Config


def initialize_data_loader(
    data_loader_type: str,
    batch_size: int,
    shuffle: bool,
    dataset_class: type[AccessLogsDataset],
    config: Config,
) -> Tuple[AccessLogsDataset, DataLoader]:
    """
    Set up a data loader.

    This function setups a data loader of a specific type (e.g., training),
    by instantiating the dataset class and creating a data loader from that
    instance.

    Args:
        data_loader_type (str): Type of data loader to create (e.g., training).
        batch_size (int): Batch size for the data loader to create.
        shuffle (bool): Whether to apply shuffle to the data loader to create.
        dataset_class (type[AccessLogsDataset]): Dataset class to instantiate.
        config (Config): Configuration object.

    Returns:
        Tuple[AccessLogsDataset, DataLoader]:
            - dataset: Instantiated dataset object of the specified class.
            - data_loader: DataLoader created from the dataset instance.
    """
    # Instantiate the dataset class
    dataset = dataset_class(data_loader_type, config)

    # Create a data loader from the
    # dataset instance
    data_loader = build_data_loader(
        dataset,
        batch_size,
        shuffle,
    )

    debug(
        "Data loader initialization executed",
        extra={
            "data_loader_type": data_loader_type,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "dataset_class": dataset_class.__name__,
            "samples_in_dataset_num": len(dataset),
            "context": "Data loader initialization",
        },
    )

    return dataset, data_loader
