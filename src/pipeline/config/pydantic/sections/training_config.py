"""training_config.py

Configuration section for the model training process.

This module structures all parameters necessary to run the training loop,
including general settings, hardware selection, and early stopping criteria.

Classes:
    TrainingDeviceConfig(BaseModel):
        Configuration for the hardware device used during training.
    TrainingGeneralConfig(BaseModel):
        General configuration for training run parameters.
    TrainingEarlyStoppingConfig(BaseModel):
        Criteria for early termination of training.
    TrainingConfig(BaseModel):
        Aggregates all training configuration settings.
"""

from pydantic import BaseModel, confloat, conint, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from const import HW_DEVICE_NAMES


class TrainingDeviceConfig(BaseModel):
    """Device configuration for training.

    Attributes:
        type (str): The device type to use.
    """

    type: str

    @model_validator(mode="after")
    def check_training_device_type(
        self: "TrainingDeviceConfig",
    ) -> "TrainingDeviceConfig":
        """Check whether training device type is valid or not.

        This function validates the training device type.

        Args:
            self (TrainingDeviceConfig): Current model instance.

        Returns:
            "TrainingDeviceConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "training.device.type",
        )

        return self


class TrainingGeneralConfig(BaseModel):
    """General training configuration.

    Attributes:
        epochs (int): Number of training epochs (> 0).
        batch_size (int): Batch size for training (> 0).
        shuffle (bool): Whether to shuffle the dataset during training.
    """

    epochs: conint(gt=0)
    batch_size: conint(gt=0)
    shuffle: bool


class TrainingEarlyStoppingConfig(BaseModel):
    """Early stopping configuration for training.

    Attributes:
        patience (int): Number of epochs with no improvement to
                        wait before stopping (>= 0).
        delta (float): Minimum change to qualify as an improvement (>= 0).
    """

    patience: conint(ge=0)
    delta: confloat(ge=0)


class TrainingConfig(BaseModel):
    """Training configuration.

    Attributes:
        device (TrainingDeviceConfig): Device configuration for training.
        general (TrainingGeneralConfig): General training configuration.
        early_stopping (TrainingEarlyStoppingConfig): Early stopping configuration.
    """

    device: TrainingDeviceConfig
    general: TrainingGeneralConfig
    early_stopping: TrainingEarlyStoppingConfig
