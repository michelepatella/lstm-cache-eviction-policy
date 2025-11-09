"""early_stopping_config.py

Configuration section for the Early Stopping mechanism.

This module defines the criteria for terminating the training loop early
based on performance metrics during the training and validation phases.

Classes:
    EarlyStoppingCriteriaConfig(BaseModel):
        Defines the patience and delta thresholds for one phase.
    EarlyStoppingConfig(BaseModel):
        Aggregates all early stopping criteria for the pipeline.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class EarlyStoppingCriteriaConfig(BaseModel):
    """Criteria for early termination of a phase.

    Attributes:
        patience (int): Number of epochs with no improvement to
                        wait before stopping (>= 0).
        delta (float): Minimum change to qualify as an improvement (>= 0).
    """

    patience: Annotated[int, Field(ge=0)]
    delta: Annotated[float, Field(ge=0)]


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration.

    Attributes:
        training (EarlyStoppingCriteriaConfig): Early stopping criteria applied
                                                to the training process.
        validation (EarlyStoppingCriteriaConfig): Early stopping criteria applied
                                                  to the validation process.
    """

    training: EarlyStoppingCriteriaConfig
    validation: EarlyStoppingCriteriaConfig
