from pydantic import BaseModel, confloat, conint, model_validator

from const import TRAINING_OPTIMIZERS
from config.classes.validation.choice_field_validator import (
    validate_choice_field,
)


# Training — General
class TrainingGeneralEpochsConfig(BaseModel):
    count: conint(gt=0)  # type: ignore[valid-type]


class TrainingGeneralConfig(BaseModel):
    epochs: TrainingGeneralEpochsConfig
    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


# Training — Optimizer — Params
class OptimizerParamsConfig(BaseModel):
    learning_rate: confloat(gt=0)  # type: ignore[valid-type]
    weight_decay: confloat(ge=0)  # type: ignore[valid-type]
    momentum: confloat(ge=0, le=1)  # type: ignore[valid-type]


# Training — Optimizer
class OptimizerConfig(BaseModel):
    type: str
    params: OptimizerParamsConfig

    @model_validator(mode="after")
    def check_data_distribution_mode(
        self: "OptimizerConfig",
    ) -> "OptimizerConfig":
        """
        Check whether optimizer is valid or not.

        This function validates the training
        optimizer specified.

        Parameters:
            self (OptimizerConfig): Current model instance.

        Returns:
            OptimizerConfig: Validated model instance.
        """
        return validate_choice_field(
            self,
            self.type,
            TRAINING_OPTIMIZERS,
            "training.optimizer.type",
        )


# Training — Early stopping
class EarlyStoppingTrainingConfig(BaseModel):
    patience: conint(ge=0)  # type: ignore[valid-type]
    delta: confloat(ge=0)  # type: ignore[valid-type]


class TrainingConfig(BaseModel):
    """
    Class representing the training configuration
    settings including general, optimizer, and
    early stopping settings.
    """

    general: TrainingGeneralConfig
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingTrainingConfig
