"""validation_config.py

Configuration section for defining model validation and
hyperparameter tuning procedures.

This module structures all parameters necessary for validation runs,
including hardware selection, general run parameters, cross-validation
settings, early stopping rules, and the full search space for
hyperparameter optimization.

Classes:
    ValidationDeviceConfig(BaseModel):
        Configuration for the hardware device used
        during validation.
    ValidationGeneralConfig(BaseModel):
        General configuration for validation runs.
    ValidationTimeSeriesCVConfig(BaseModel):
        Time Series Cross-Validation (TSCV) parameters.
    ValidationEarlyStoppingConfig(BaseModel):
        Criteria for early termination during validation.
    ValidationSearchSpaceModelConfig(BaseModel):
        Search space for model hyperparameters.
    ValidationSearchSpaceOptimizerConfig(BaseModel):
        Search space for optimizer.
    ValidationSearchSpaceConfig(BaseModel):
        Aggregates all hyperparameter search spaces.
    ValidationConfig(BaseModel):
        Root class aggregating all validation settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import HW_DEVICE_NAMES


class ValidationDeviceConfig(BaseModel):
    """Device configuration for validation.

    Attributes:
        type (str): The device type to use.
    """

    type: str

    @model_validator(mode="after")
    def check_validation_device_type(
        self: "ValidationDeviceConfig",
    ) -> "ValidationDeviceConfig":
        """Check whether validation device type is valid or not.

        This function validates the validation device type.

        Args:
            self (ValidationDeviceConfig): Current model instance.

        Returns:
            "ValidationDeviceConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "validation.device.type",
        )

        return self


class ValidationGeneralConfig(BaseModel):
    """General validation configuration.

    Attributes:
        batch_size (int): Batch size for validation (> 0).
        shuffle (bool): Whether to shuffle the dataset during validation.
    """

    batch_size: Annotated[int, Field(gt=0)]
    shuffle: bool


class ValidationTimeSeriesCVConfig(BaseModel):
    """Time Series Cross-validation configuration.

    Attributes:
        folds (int): Number of folds for cross-validation (> 1).
        epochs (int): Number of epochs for each fold (> 0).
    """

    folds: Annotated[int, Field(gt=1)]
    epochs: Annotated[int, Field(gt=0)]


class ValidationEarlyStoppingConfig(BaseModel):
    """Early stopping configuration for validation.

    Attributes:
        patience (int): Number of epochs with no improvement to wait
                        before stopping (>= 0).
        delta (float): Minimum change to qualify as improvement (>= 0).
    """

    patience: Annotated[int, Field(ge=0)]
    delta: Annotated[float, Field(ge=0)]


class ValidationSearchSpaceModelConfig(BaseModel):
    """Model search space configuration for hyperparameter optimization.

    Attributes:
        hidden_size (list[int]): Allowed range of hidden layer sizes (> 0).
        num_layers (list[int]): Allowed range of number of layers (> 0).
        dropout (list[float]): Allowed range of dropout values (in [0, 1)).
    """

    hidden_size: list[Annotated[int, Field(gt=0)]]
    num_layers: list[Annotated[int, Field(gt=0)]]
    dropout: list[Annotated[float, Field(ge=0, lt=1)]]


class ValidationSearchSpaceOptimizerConfig(BaseModel):
    """Optimizer search space configuration for hyperparameter tuning.

    Attributes:
        learning_rate (list[float]): Allowed learning rate values (> 0).
    """

    learning_rate: list[Annotated[float, Field(gt=0)]]


class ValidationSearchSpaceConfig(BaseModel):
    """Search space configuration.

    Attributes:
        model (ValidationSearchSpaceModelConfig): Model hyperparameter search space
                                        configuration.
        training (ValidationSearchSpaceOptimizerConfig): Training hyperparameter search space
                                                         configuration.
    """

    model: ValidationSearchSpaceModelConfig
    optimizer: ValidationSearchSpaceOptimizerConfig


class ValidationConfig(BaseModel):
    """Validation configuration.

    Attributes:
        device (ValidationDeviceConfig): Device configuration for validation.
        general (ValidationGeneralConfig): General validation configuration.
        time_series_cv (ValidationTimeSeriesCVConfig): Time series
                                                       cross-validation configuration.
        early_stopping (ValidationEarlyStoppingConfig): Early stopping configuration.
        search_space (ValidationSearchSpaceConfig): Hyperparameter search space
                                                    configuration for tuning.
    """

    device: ValidationDeviceConfig
    general: ValidationGeneralConfig
    time_series_cv: ValidationTimeSeriesCVConfig
    early_stopping: ValidationEarlyStoppingConfig
    search_space: ValidationSearchSpaceConfig
