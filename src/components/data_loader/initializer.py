"""initializer.py

Helper module for initializing PyTorch DataLoaders with datasets.

This module provides the `initialize_data_loader` function, which instantiates
a dataset of the specified class and type, then builds a DataLoader with the
given batch size and shuffle configuration.

Functions:
    initialize_data_loader(
        dataset_type: str,
        split_type: str,
        batch_size: int,
        shuffle: bool,
        dataset_class: Type[AccessLogsDataset],
        pipeline_config: PipelineConfig
    ) -> tuple[AccessLogsDataset, DataLoader]
        Instantiates the dataset and returns it along with a configured DataLoader.
"""

from torch.utils.data import DataLoader

from components.data_loader.builder import build_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.logs.levels.debug_logger import debug
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def initialize_data_loader(
    dataset_type: str,
    split_type: str,
    batch_size: int,
    shuffle: bool,
    dataset_class: type[AccessLogsDataset],
    pipeline_config: PipelineConfig,
) -> tuple[AccessLogsDataset, DataLoader]:
    """Set up a data loader.

    This function setups a data loader of a specific type,
    by instantiating the dataset class and creating a data loader from that
    instance.

    Args:
        dataset_type (str): Type of the dataset requested.
        split_type (str): Type of split to apply.
        batch_size (int): Batch size for the data loader to create.
        shuffle (bool): Whether to apply shuffle to the data loader to create.
        dataset_class (type[AccessLogsDataset]): Dataset class to instantiate.
        pipeline_config (PipelineConfig): Configuration object.

    Returns:
        tuple[AccessLogsDataset, DataLoader]:
            - dataset: Instantiated dataset object of the specified class.
            - data_loader: DataLoader created from the dataset instance.
    """
    # Instantiate the dataset class
    dataset = dataset_class(dataset_type, split_type, pipeline_config)

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
            "data_loader_type": split_type,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "dataset_class": dataset_class.__name__,
            "samples_in_dataset_num": len(dataset),
            "context": "Data loader initialization",
        },
    )

    return dataset, data_loader
