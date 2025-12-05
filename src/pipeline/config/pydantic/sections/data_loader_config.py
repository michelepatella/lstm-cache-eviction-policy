"""data_loader_config.py

Configuration section for the data loader.

This module structures all parameters necessary for creating DataLoader objects,
including batch sizing and shuffling settings for different phases (training,
validation, testing).

Classes:
    DataLoaderBatchSizeConfig(BaseModel):
        Defines the batch size for each phase.
    DataLoaderShuffleConfig(BaseModel):
        Defines whether to shuffle the dataset for each phase.
    DataLoaderConfig(BaseModel):
        Aggregates all data loader configuration settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class DataLoaderBatchSizeConfig(BaseModel):
    """Batch size configuration for the data loader.

    Attributes:
        training (int): Batch size used during the training phase (> 0).
        validation (int): Batch size used during the validation phase (> 0).
        testing (int): Batch size used during the inference (> 0).
    """

    training: Annotated[int, Field(gt=0)]
    validation: Annotated[int, Field(gt=0)]
    testing: Annotated[int, Field(gt=0)]


class DataLoaderShuffleConfig(BaseModel):
    """Shuffle configuration for data loader.

    Attributes:
        training (bool): Whether to shuffle the training dataset.
        validation (bool): Whether to shuffle the validation dataset.
        testing (bool): Whether to shuffle the testing dataset.
    """

    training: bool
    validation: bool
    testing: bool


class DataLoaderConfig(BaseModel):
    """Data loader configuration.

    Attributes:
        batch_size (DataLoaderBatchSizeConfig): Batch size settings for different
                                                phases.
        shuffle (DataLoaderShuffleConfig): Shuffle settings for different phases.
    """

    batch_size: DataLoaderBatchSizeConfig
    shuffle: DataLoaderShuffleConfig
