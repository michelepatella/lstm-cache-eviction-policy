from pydantic import (
    BaseModel,
    confloat,
    conint,
    model_validator,
)

from const import ALLOWED_TRAINING_OPTIMIZERS
from lstm_eviction_policy.config.classes.validation.choice_field_validator import (
    validate_choice_field,
)


# Training — General
class TrainingGeneralEpochsConfig(BaseModel):
    count: conint(gt=0)


class TrainingGeneralConfig(BaseModel):
    epochs: TrainingGeneralEpochsConfig
    batch_size: conint(gt=0)


# Training — Optimizer — Params
class OptimizerParamsConfig(BaseModel):
    learning_rate: confloat(gt=0)
    weight_decay: confloat(ge=0)
    momentum: confloat(ge=0, le=1)


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
            ALLOWED_TRAINING_OPTIMIZERS,
            "training.optimizer.type",
        )


# Training — Early stopping
class EarlyStoppingTrainingConfig(BaseModel):
    patience: conint(ge=0)
    delta: confloat(ge=0)


class TrainingConfig(BaseModel):
    """
    Class representing the training configuration
    settings including general, optimizer, and
    early stopping settings.
    """

    general: TrainingGeneralConfig
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingTrainingConfig
