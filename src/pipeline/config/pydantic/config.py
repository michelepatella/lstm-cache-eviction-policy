"""config.py

Main Pydantic model aggregating all configuration sections for
the pipeline.

This module provides the Config class, which is the root structure for
loading and validating the entire system configuration from a YAML file.

Classes:
    Config: Represents the full configuration of the system.
"""

from pydantic import BaseModel

# Importazioni dei moduli Pydantic ancora da definire (placeholder):
# In una configurazione reale, ogni chiave YAML deve avere la sua classe importata
# Sostituiamo le classi generiche con i nomi che corrispondono al YAML:
# (ASSUMENDO che i file esistano e contengano le classi con i nomi standard)
from pipeline.config.pydantic.sections.data_config import DataConfig

# Importazioni dei moduli Pydantic creati in precedenza:
from pipeline.config.pydantic.sections.data_loader_config import (
    DataLoaderConfig,
)
from pipeline.config.pydantic.sections.dataset_config import DatasetConfig
from pipeline.config.pydantic.sections.early_stopping_config import (
    EarlyStoppingConfig,
)
from pipeline.config.pydantic.sections.evaluation_config import (
    EvaluationConfig,
)
from pipeline.config.pydantic.sections.logs_config import LogsConfig
from pipeline.config.pydantic.sections.loss_config import LossConfig
from pipeline.config.pydantic.sections.model_config import ModelConfig
from pipeline.config.pydantic.sections.optimizer_config import OptimizerConfig
from pipeline.config.pydantic.sections.resources_config import (
    ResourcesConfig,
)
from pipeline.config.pydantic.sections.seed_config import SeedConfig
from pipeline.config.pydantic.sections.simulations_config import (
    SimulationsConfig,
)
from pipeline.config.pydantic.sections.training_config import TrainingConfig
from pipeline.config.pydantic.sections.validation_config import (
    ValidationConfig,
)


class Config(BaseModel):
    """Represents the full configuration of the system.

    This class aggregates all configuration sections used across
    the system, providing structured access and validation against
    the global YAML configuration file.

    Attributes:
        data (DataConfig): Data configuration.
        dataset (DatasetConfig): Dataset configuration.
        data_loader (DataLoaderConfig): Dataloader configuration.
        early_stopping (EarlyStoppingConfig): Early stopping configuration.
        evaluation (EvaluationConfig): Evaluation configuration.
        logs (LogsConfig): Logs configuration.
        loss (LossConfig): Loss configuration.
        model (ModelConfig): Model configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        resources (ResourcesConfig): Resources configuration.
        seed (SeedConfig): Seed configuration.
        simulations (SimulationsConfig): Simulations configuration.
        training (TrainingConfig): Training configuration.
        validation (ValidationConfig): Validation configuration.
    """

    data: DataConfig
    dataset: DatasetConfig
    data_loader: DataLoaderConfig
    early_stopping: EarlyStoppingConfig
    evaluation: EvaluationConfig
    logs: LogsConfig
    loss: LossConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    resources: ResourcesConfig
    seed: SeedConfig
    simulations: SimulationsConfig
    training: TrainingConfig
    validation: ValidationConfig
