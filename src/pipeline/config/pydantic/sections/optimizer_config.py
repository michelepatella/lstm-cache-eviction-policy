"""optimizer_config.py

Configuration section for the model's primary optimizer settings.

This module defines the type of optimizer to use and its key
hyperparameters, such as learning rate, momentum, and weight decay.

Classes:
    OptimizerParamsConfig(BaseModel):
        Configuration for optimizer hyperparameters.
    OptimizerConfig(BaseModel):
        Aggregates the optimizer type and its parameters.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from const import OPTIMIZER_NAMES


class OptimizerParamsConfig(BaseModel):
    """Configuration for optimizer hyperparameters.

    Attributes:
        learning_rate (float): The learning rate (> 0).
        momentum (float): The momentum factor (in [0,1]).
        weight_decay (float): The weight decay factor (>= 0).
    """

    learning_rate: Annotated[float, Field(gt=0)]
    momentum: Annotated[float, Field(ge=0, le=1)]
    weight_decay: Annotated[float, Field(ge=0)]


class OptimizerConfig(BaseModel):
    """Aggregates the optimizer type and its parameters.

    Attributes:
        params (OptimizerParamsConfig): Optimizer hyperparameters.
        type (str): The optimizer type to be used.
    """

    params: OptimizerParamsConfig
    type: str

    @model_validator(mode="after")
    def check_optimizer_type(
        self: "OptimizerConfig",
    ) -> "OptimizerConfig":
        """Check whether the optimizer type is valid or not.

        This function validates the specified optimizer type.

        Args:
            self (OptimizerConfig): Current model instance.

        Returns:
            "OptimizerConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            OPTIMIZER_NAMES,
            "optimizer.type",
        )

        return self
