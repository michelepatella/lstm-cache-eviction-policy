"""validation_pipeline_config.py

Configuration section for defining model validation and
hyperparameter tuning procedures.

This module structures all parameters necessary for validation runs,
focusing on Time Series Cross-Validation (TSCV) settings and the
full search space for hyperparameter optimization.

Classes:
    ValidationTimeSeriesCVPipelineConfig(BaseModel):
        Time Series Cross-Validation (TSCV) parameters.
    ValidationSearchSpaceModelPipelineConfig(BaseModel):
        Search space for model hyperparameters.
    ValidationSearchSpaceOptimizerPipelineConfig(BaseModel):
        Search space for optimizer.
    ValidationSearchSpacePipelineConfig(BaseModel):
        Aggregates all hyperparameter search spaces.
    ValidationPipelineConfig(BaseModel):
        Root class aggregating TSCV and search space settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ValidationTimeSeriesCVPipelineConfig(BaseModel):
    """Time Series Cross-validation configuration.

    Attributes:
        folds (int): Number of folds for cross-validation (> 1).
        epochs (int): Number of epochs for each fold (> 0).
    """

    folds: Annotated[int, Field(gt=1)]
    epochs: Annotated[int, Field(gt=0)]


class ValidationSearchSpaceModelPipelineConfig(BaseModel):
    """Model search space configuration for hyperparameter optimization.

    Attributes:
        hidden_size (list[int]): Allowed range of hidden layer sizes (> 0).
        num_layers (list[int]): Allowed range of number of layers (> 0).
        dropout (list[float]): Allowed range of dropout values
                               (in [0.0, 1.0]).
    """

    hidden_size: list[Annotated[int, Field(gt=0)]]
    num_layers: list[Annotated[int, Field(gt=0)]]
    dropout: list[Annotated[float, Field(ge=0, le=1)]]


class ValidationSearchSpaceOptimizerPipelineConfig(BaseModel):
    """Optimizer search space configuration for hyperparameter tuning.

    Attributes:
        learning_rate (list[float]): Allowed learning rate values (> 0).
    """

    learning_rate: list[Annotated[float, Field(gt=0)]]


class ValidationSearchSpacePipelineConfig(BaseModel):
    """Search space configuration.

    Attributes:
        model (ValidationSearchSpaceModelPipelineConfig): Model hyperparameter
                                                          search space configuration.
        optimizer (ValidationSearchSpaceOptimizerPipelineConfig):
            Optimizer hyperparameter search space configuration.
    """

    model: ValidationSearchSpaceModelPipelineConfig
    optimizer: ValidationSearchSpaceOptimizerPipelineConfig


class ValidationPipelineConfig(BaseModel):
    """Validation configuration.

    Attributes:
        time_series_cv (ValidationTimeSeriesCVPipelineConfig):
            Time series cross-validation configuration.
        search_space (ValidationSearchSpacePipelineConfig):
            Hyperparameter search space configuration for tuning.
    """

    time_series_cv: ValidationTimeSeriesCVPipelineConfig
    search_space: ValidationSearchSpacePipelineConfig
