from torch.utils.data import DataLoader

from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.data.dataloader.data_loader_builder import (
    create_data_loader,
)
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def data_loader_setup(
    data_loader_type: str,
    batch_size: int,
    shuffle: bool,
    config: Config,
    dataset_class: type[AccessLogsDataset],
) -> tuple[AccessLogsDataset, DataLoader]:
    """
    Set up a data loader for further usage.

    This function setups a data loader
    of a specific type (e.g. training), by
    instantiating the dataset class and creating
    a dataloader from the instance.

    Parameters:
        data_loader_type (str): Type of data loader to create.
        batch_size (int): Batch size for the data loader to create.
        shuffle (bool): Whether to apply shuffle to the data loader.
        config (Config): Configuration object.
        dataset_class (type[AccessLogsDataset]): Dataset class to instantiate
                                                 and from which to create the
                                                 data loader.

    Returns:
        tuple[AccessLogsDataset, DataLoader]: Tuple of dataset instance
                                              and data loader created.
    """
    # Get a dataset instance
    dataset = dataset_class(data_loader_type, config)

    # Create a data loader from the dataset
    data_loader = create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    info("Data loader setup completed")

    return dataset, data_loader
