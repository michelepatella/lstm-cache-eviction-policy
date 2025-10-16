from typing import Tuple

from torch.utils.data import DataLoader

from components.data_loader.builder import build_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.info_logger import info
from pipeline.config.pydantic.config import Config


def initialize_data_loader(
    data_loader_type: str,
    batch_size: int,
    shuffle: bool,
    dataset_class: type[AccessLogsDataset],
    config: Config,
) -> Tuple[AccessLogsDataset, DataLoader]:
    """
    Set up a data loader for further usage.

    This function setups a data loader
    of a specific type (e.g. training), by
    instantiating the dataset class and creating
    a dataloader from the instance.

    Args:
        data_loader_type (str): Type of data loader to create.
        batch_size (int): Batch size for the data loader to create.
        shuffle (bool): Whether to apply shuffle to the data loader.
        config (Config): Configuration object.
        dataset_class (type[AccessLogsDataset]): Dataset class to instantiate
                                                 and from which to create the
                                                 data loader.

    Returns:
        Tuple[AccessLogsDataset, DataLoader]:
            - dataset: Instantiated dataset object of the specified class.
            - data_loader: DataLoader created from the dataset instance.
    """
    # Get a dataset instance
    dataset = dataset_class(data_loader_type, config)

    # Create a data loader from the dataset
    data_loader = build_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    info("Data loader initialization completed")

    return dataset, data_loader
