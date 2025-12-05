"""training_config.py

Configuration section for the training phase.

Classes:
    TrainingEpochsConfig(BaseModel):
        Configuration for the training phase.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration.

    Attributes:
        epochs (int): Number of training epochs (> 0).
    """

    epochs: Annotated[int, Field(gt=0)]
