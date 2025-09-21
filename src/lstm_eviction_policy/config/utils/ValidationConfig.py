from typing import List

from pydantic import BaseModel, confloat, conint


# Validation — Cross validation — Folds & Epochs
class FoldsConfig(BaseModel):
    count: conint(gt=1)


class EpochsConfig(BaseModel):
    count: conint(gt=0)


class CrossValidationConfig(BaseModel):
    folds: FoldsConfig
    epochs: EpochsConfig


# Validation — Early stopping
class EarlyStoppingValidationConfig(BaseModel):
    patience: conint(ge=0)
    delta: confloat(ge=0)


# Validation — Search space — Model
class ModelSearchSpaceConfig(BaseModel):
    hidden_size_range: List[conint(gt=0)]
    num_layers_range: List[conint(gt=0)]
    dropout_range: List[confloat(ge=0, lt=1)]


# Validation — Search space — Training
class TrainingSearchSpaceOptimizerConfig(BaseModel):
    learning_rate_range: List[confloat(gt=0)]


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

    cross_validation: CrossValidationConfig
    early_stopping: EarlyStoppingValidationConfig
    search_space: SearchSpaceConfig
