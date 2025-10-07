from typing import List

from pydantic import BaseModel, confloat, conint


# Validation — General
class GeneralValidationConfig(BaseModel):
    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


# Validation — Cross validation
class CrossValidationConfig(BaseModel):
    folds: conint(gt=1)  # type: ignore[valid-type]
    epochs: conint(gt=0)  # type: ignore[valid-type]


# Validation — Early stopping
class EarlyStoppingValidationConfig(BaseModel):
    patience: conint(ge=0)  # type: ignore[valid-type]
    delta: confloat(ge=0)  # type: ignore[valid-type]


# Validation — Search space — Model
class ModelSearchSpaceConfig(BaseModel):
    hidden_size_range: List[conint(gt=0)]  # type: ignore[valid-type]
    num_layers_range: List[conint(gt=0)]  # type: ignore[valid-type]
    dropout_range: List[confloat(ge=0, lt=1)]  # type: ignore[valid-type]


# Validation — Search space — Training
class TrainingSearchSpaceOptimizerConfig(BaseModel):
    learning_rate_range: List[confloat(gt=0)]  # type: ignore[valid-type]


class TrainingSearchSpaceConfig(BaseModel):
    optimizer: TrainingSearchSpaceOptimizerConfig


# Validation — Search space
class SearchSpaceConfig(BaseModel):
    model: ModelSearchSpaceConfig
    training: TrainingSearchSpaceConfig


class ValidationConfig(BaseModel):
    """
    Class representing the validation configuration
    settings including cross validation, early
    stopping, and search space settings.
    """

    general: GeneralValidationConfig
    cross_validation: CrossValidationConfig
    early_stopping: EarlyStoppingValidationConfig
    search_space: SearchSpaceConfig
