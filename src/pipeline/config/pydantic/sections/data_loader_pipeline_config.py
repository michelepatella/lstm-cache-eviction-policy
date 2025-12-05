"""data_loader_pipeline_config.py

Configuration section for the data loader.

This module structures all parameters necessary for creating DataLoader objects,
including batch sizing and shuffling settings for different phases (training,
validation, testing).

Classes:
    DataLoaderPhaseConfig(BaseModel):
        Defines the batch size and shuffle setting for a single phase.
    DataLoaderPipelineConfig(BaseModel):
        Aggregates all data loader configuration settings by phase.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class DataLoaderPhaseConfig(BaseModel):
    """Configuration for a single data loading phase (training, validation,
    or testing).

    Attributes:
        batch_size (int): Batch size used during the phase (> 0).
        shuffle (bool): Whether to shuffle the dataset during the phase.
    """

    batch_size: Annotated[int, Field(gt=0)]
    shuffle: bool


class DataLoaderPipelineConfig(BaseModel):
    """Data loader configuration, structured by phase.

    Attributes:
        training (DataLoaderPhaseConfig): Configuration for the training phase.
        validation (DataLoaderPhaseConfig): Configuration for the validation phase.
        testing (DataLoaderPhaseConfig): Configuration for the testing phase (inference).
    """

    training: DataLoaderPhaseConfig
    validation: DataLoaderPhaseConfig
    testing: DataLoaderPhaseConfig
