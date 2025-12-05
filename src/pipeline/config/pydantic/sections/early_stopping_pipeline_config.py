"""early_stopping_pipeline_config.py

Configuration section for the Early Stopping mechanism.

This module defines the criteria for terminating the training loop early
based on performance metrics during the training and validation phases.

Classes:
    EarlyStoppingCriteriaPipelineConfig(BaseModel):
        Defines the patience and delta thresholds for one phase.
    EarlyStoppingPipelineConfig(BaseModel):
        Aggregates all early stopping criteria for the pipeline.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class EarlyStoppingCriteriaPipelineConfig(BaseModel):
    """Criteria for early termination of a phase.

    Attributes:
        patience (int): Number of epochs with no improvement to
                        wait before stopping (>= 0).
        delta (float): Minimum change to qualify as an improvement (>= 0).
    """

    patience: Annotated[int, Field(ge=0)]
    delta: Annotated[float, Field(ge=0)]


class EarlyStoppingPipelineConfig(BaseModel):
    """Early stopping configuration.

    Attributes:
        training (EarlyStoppingCriteriaPipelineConfig): Early stopping criteria
                                                        applied to the training
                                                        process.
        validation (EarlyStoppingCriteriaPipelineConfig): Early stopping criteria
                                                          applied to the validation
                                                          process.
    """

    training: EarlyStoppingCriteriaPipelineConfig
    validation: EarlyStoppingCriteriaPipelineConfig
