from pydantic import BaseModel, confloat, conint, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from const import OPTIMIZER_NAMES


class TrainingGeneralConfig(BaseModel):
    """
    General training configuration.

    Attributes:
        epochs (int): Number of training epochs (> 0).
        batch_size (int): Batch size for training (> 0).
        shuffle (bool): Whether to shuffle the dataset during training.
    """

    epochs: conint(gt=0)  # type: ignore[valid-type]
    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


class OptimizerParamsConfig(BaseModel):
    """
    Configuration for the training optimizer.

    Attributes:
        learning_rate (float): Learning rate (> 0).
        weight_decay (float): Weight decay (>= 0).
        momentum (float): Momentum (in [0,1]).
    """

    learning_rate: confloat(gt=0)  # type: ignore[valid-type]
    weight_decay: confloat(ge=0)  # type: ignore[valid-type]
    momentum: confloat(ge=0, le=1)  # type: ignore[valid-type]


class OptimizerConfig(BaseModel):
    """
    Optimizer configuration for training.

    Attributes:
        type (str): Optimizer type.
        params (OptimizerParamsConfig): Configuration for the optimizer.
    """

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

        Args:
            self (OptimizerConfig): Current model instance.

        Returns:
            "OptimizerConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            OPTIMIZER_NAMES,
            "training.optimizer.type",
        )

        return self


class EarlyStoppingTrainingConfig(BaseModel):
    """
    Early stopping configuration for training.

    Attributes:
        patience (int): Number of epochs with no improvement to
                        wait before stopping (>= 0).
        delta (float): Minimum change to qualify as an improvement (>= 0).
    """

    patience: conint(ge=0)  # type: ignore[valid-type]
    delta: confloat(ge=0)  # type: ignore[valid-type]


class TrainingConfig(BaseModel):
    """
    Training configuration.

    Attributes:
        general (TrainingGeneralConfig): General training configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        early_stopping (EarlyStoppingTrainingConfig): Early stopping configuration.
    """

    general: TrainingGeneralConfig
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingTrainingConfig
