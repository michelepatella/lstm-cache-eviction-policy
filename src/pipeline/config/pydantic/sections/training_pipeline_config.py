"""training_pipeline_config.py

Configuration section for the training phase.

Classes:
    TrainingEpochsPipelineConfig(BaseModel):
        Configuration for the training phase.
    TrainingSamplesPipelineConfig(BaseModel):
        Configuration for the training samples.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class TrainingSamplesPipelineConfig(BaseModel):
    """Training samples configuration.

    Attributes:
        min (int): Minimum number of samples for training process.
    """

    min: Annotated[int, Field(gt=0)]


class TrainingPipelineConfig(BaseModel):
    """Training configuration.

    Attributes:
        epochs (int): Number of training epochs (> 0).
    """

    epochs: Annotated[int, Field(gt=0)]
    samples: TrainingSamplesPipelineConfig
