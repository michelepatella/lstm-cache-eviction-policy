from pydantic import BaseModel, confloat, conint


class ValidationGeneralConfig(BaseModel):
    """General validation configuration.

    Attributes:
        batch_size (int): Batch size for validation (> 0).
        shuffle (bool): Whether to shuffle the dataset during validation.
    """

    batch_size: conint(gt=0)
    shuffle: bool


class ValidationCrossValidationConfig(BaseModel):
    """Cross-validation configuration.

    Attributes:
        folds (int): Number of folds for cross-validation (> 1).
        epochs (int): Number of epochs for each fold (> 0).
    """

    folds: conint(gt=1)
    epochs: conint(gt=0)


class ValidationEarlyStoppingConfig(BaseModel):
    """Early stopping configuration for validation.

    Attributes:
        patience (int): Number of epochs with no improvement to wait
                        before stopping (>= 0).
        delta (float): Minimum change to qualify as improvement (>= 0).
    """

    patience: conint(ge=0)
    delta: confloat(ge=0)


class ValidationSearchSpaceModelConfig(BaseModel):
    """Model search space configuration for hyperparameter optimization.

    Attributes:
        hidden_size (list[int]): Allowed range of hidden layer sizes (> 0).
        num_layers (list[int]): Allowed range of number of layers (> 0).
        dropout (list[float]): Allowed range of dropout values (in [0, 1)).
    """

    hidden_size: list[conint(gt=0)]  # type: ignore[valid-type]
    num_layers: list[conint(gt=0)]  # type: ignore[valid-type]
    dropout: list[confloat(ge=0, lt=1)]  # type: ignore[valid-type]


class ValidationSearchSpaceTrainingOptimizerConfig(BaseModel):
    """Optimizer search space configuration for hyperparameter tuning.

    Attributes:
        learning_rate (list[float]): Allowed learning rate values (> 0).
    """

    learning_rate: list[confloat(gt=0)]  # type: ignore[valid-type]


class ValidationSearchSpaceTrainingConfig(BaseModel):
    """Training search space configuration.

    Attributes:
        optimizer (ValidationSearchSpaceTrainingOptimizerConfig): Optimizer search space
                                                        configuration.
    """

    optimizer: ValidationSearchSpaceTrainingOptimizerConfig


class ValidationSearchSpaceConfig(BaseModel):
    """Search space configuration.

    Attributes:
        model (ValidationSearchSpaceModelConfig): Model hyperparameter search space
                                        configuration.
        training (ValidationSearchSpaceTrainingConfig): Training hyperparameter search space
                                              configuration.
    """

    model: ValidationSearchSpaceModelConfig
    training: ValidationSearchSpaceTrainingConfig


class ValidationConfig(BaseModel):
    """Validation configuration.

    Attributes:
        general (ValidationGeneralConfig): General validation configuration.
        cross_validation (ValidationCrossValidationConfig): Cross-validation configuration.
        early_stopping (ValidationEarlyStoppingConfig): Early stopping configuration.
        search_space (ValidationSearchSpaceConfig): Hyperparameter search space configuration
                                          for tuning.
    """

    general: ValidationGeneralConfig
    cross_validation: ValidationCrossValidationConfig
    early_stopping: ValidationEarlyStoppingConfig
    search_space: ValidationSearchSpaceConfig
