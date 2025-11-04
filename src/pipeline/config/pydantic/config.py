"""config.py

Main Pydantic model aggregating all configuration sections for
the pipeline.

This module provides the Config class, which is the root structure for
loading and validating the entire system configuration from a YAML file.

Classes:
    Config: Represents the full configuration of the system.
"""

from pydantic import BaseModel

from pipeline.config.pydantic.sections.api_kwargs_config import (
    ApiKwargsConfig,
)
from pipeline.config.pydantic.sections.caches_config import CachesConfig
from pipeline.config.pydantic.sections.data_config import DataConfig
from pipeline.config.pydantic.sections.dataset_config import DatasetConfig
from pipeline.config.pydantic.sections.evaluation_config import (
    EvaluationConfig,
)
from pipeline.config.pydantic.sections.logs_config import LogsConfig
from pipeline.config.pydantic.sections.loss_config import LossConfig
from pipeline.config.pydantic.sections.model_config import ModelConfig
from pipeline.config.pydantic.sections.optimizer_config import OptimizerConfig
from pipeline.config.pydantic.sections.testing_config import TestingConfig
from pipeline.config.pydantic.sections.training_config import TrainingConfig
from pipeline.config.pydantic.sections.validation_config import (
    ValidationConfig,
)


class Config(BaseModel):
    """Represents the full configuration of the system.

    This class aggregates all configuration sections used across
    the system, providing structured access to hardware, data,
    dataset, model, validation, training, testing, and simulations
    configurations.

    Attributes:
        api_kwargs (ApiKwargsConfig): API-specific arguments.
        caches (CachesConfig): General cache configuration parameters.
        data (DataConfig): Data configuration.
        dataset (DatasetConfig): Dataset configuration.
        evaluation (EvaluationConfig): Evaluation metrics configuration.
        logs (LogsConfig): Logging configuration.
        loss (LossConfig): Loss function configuration.
        model (ModelConfig): Model configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        testing (TestingConfig): Testing configuration.
        training (TrainingConfig): Training configuration.
        validation (ValidationConfig): Validation configuration.
    """

    api_kwargs: ApiKwargsConfig
    caches: CachesConfig
    data: DataConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    logs: LogsConfig
    loss: LossConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    testing: TestingConfig
    training: TrainingConfig
    validation: ValidationConfig
