from typing import List

from pydantic import BaseModel, confloat, conint


class GeneralValidationConfig(BaseModel):
    """
    General validation configuration.

    Attributes:
        batch_size (int): Batch size for validation (> 0).
        shuffle (bool): Whether to shuffle the dataset during validation.
    """

    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


class CrossValidationConfig(BaseModel):
    """
    Cross-validation configuration.

    Attributes:
        folds (int): Number of folds for cross-validation (> 1).
        epochs (int): Number of epochs for each fold (> 0).
    """

    folds: conint(gt=1)  # type: ignore[valid-type]
    epochs: conint(gt=0)  # type: ignore[valid-type]


class EarlyStoppingValidationConfig(BaseModel):
    """
    Early stopping configuration for validation.

    Attributes:
        patience (int): Number of epochs with no improvement to wait
                        before stopping (>= 0).
        delta (float): Minimum change to qualify as improvement (>= 0).
    """

    patience: conint(ge=0)  # type: ignore[valid-type]
    delta: confloat(ge=0)  # type: ignore[valid-type]


class ModelSearchSpaceConfig(BaseModel):
    """
    Model search space configuration for hyperparameter optimization.

    Attributes:
        hidden_size (List[int]): Allowed range of hidden layer sizes (> 0).
        num_layers (List[int]): Allowed range of number of layers (> 0).
        dropout (List[float]): Allowed range of dropout values (in [0, 1)).
    """

    hidden_size: List[conint(gt=0)]  # type: ignore[valid-type]
    num_layers: List[conint(gt=0)]  # type: ignore[valid-type]
    dropout: List[confloat(ge=0, lt=1)]  # type: ignore[valid-type]


class TrainingSearchSpaceOptimizerConfig(BaseModel):
    """
    Optimizer search space configuration for hyperparameter tuning.

    Attributes:
        learning_rate (List[float]): Allowed learning rate values (> 0).
    """

    learning_rate: List[confloat(gt=0)]  # type: ignore[valid-type]


class TrainingSearchSpaceConfig(BaseModel):
    """
    Training search space configuration.

    Attributes:
        optimizer (TrainingSearchSpaceOptimizerConfig): Optimizer search space
                                                        configuration.
    """

    optimizer: TrainingSearchSpaceOptimizerConfig


class SearchSpaceConfig(BaseModel):
    """
    Search space configuration.

    Attributes:
        model (ModelSearchSpaceConfig): Model hyperparameter search space
                                        configuration.
        training (TrainingSearchSpaceConfig): Training hyperparameter search space
                                              configuration.
    """

    model: ModelSearchSpaceConfig
    training: TrainingSearchSpaceConfig


class ValidationConfig(BaseModel):
    """
    Validation configuration.

    Attributes:
        general (GeneralValidationConfig): General validation configuration.
        cross_validation (CrossValidationConfig): Cross-validation configuration.
        early_stopping (EarlyStoppingValidationConfig): Early stopping configuration.
        search_space (SearchSpaceConfig): Hyperparameter search space configuration
                                          for tuning.
    """

    general: GeneralValidationConfig
    cross_validation: CrossValidationConfig
    early_stopping: EarlyStoppingValidationConfig
    search_space: SearchSpaceConfig
