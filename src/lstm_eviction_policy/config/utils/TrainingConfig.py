from pydantic import BaseModel, confloat, conint


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
