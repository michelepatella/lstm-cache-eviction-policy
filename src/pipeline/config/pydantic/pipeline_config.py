"""pipeline_config.py

Main Pydantic model aggregating all configuration sections for
the pipeline.

This module provides the Config class, which is the root structure for
loading and validating the entire system configuration from a YAML file.

Classes:
    PipelineConfig: Represents the full configuration of the pipeline.
"""

from pydantic import BaseModel

# Importazioni dei moduli Pydantic creati in precedenza:
from pipeline.config.pydantic.sections.data_loader_pipeline_config import (
    DataLoaderPipelineConfig,
)

# Importazioni dei moduli Pydantic ancora da definire (placeholder):
# In una configurazione reale, ogni chiave YAML deve avere la sua classe importata
# Sostituiamo le classi generiche con i nomi che corrispondono al YAML:
# (ASSUMENDO che i file esistano e contengano le classi con i nomi standard)
from pipeline.config.pydantic.sections.data_pipeline_config import (
    DataPipelineConfig,
)
from pipeline.config.pydantic.sections.dataset_pipeline_config import (
    DatasetPipelineConfig,
)
from pipeline.config.pydantic.sections.early_stopping_pipeline_config import (
    EarlyStoppingPipelineConfig,
)
from pipeline.config.pydantic.sections.evaluation_pipeline_config import (
    EvaluationPipelineConfig,
)
from pipeline.config.pydantic.sections.logs_pipeline_config import (
    LogsPipelineConfig,
)
from pipeline.config.pydantic.sections.loss_pipeline_config import (
    LossPipelineConfig,
)
from pipeline.config.pydantic.sections.model_pipeline_config import (
    ModelPipelineConfig,
)
from pipeline.config.pydantic.sections.optimizer_pipeline_config import (
    OptimizerPipelineConfig,
)
from pipeline.config.pydantic.sections.resources_pipeline_config import (
    ResourcesPipelineConfig,
)
from pipeline.config.pydantic.sections.seed_pipeline_config import (
    SeedPipelineConfig,
)
from pipeline.config.pydantic.sections.simulations_pipeline_config import (
    SimulationsPipelineConfig,
)
from pipeline.config.pydantic.sections.training_pipeline_config import (
    TrainingPipelineConfig,
)
from pipeline.config.pydantic.sections.validation_pipeline_config import (
    ValidationPipelineConfig,
)


class PipelineConfig(BaseModel):
    """Represents the full configuration of the pipeline.

    This class aggregates all configuration sections used across
    the system, providing structured access and validation against
    the global YAML configuration file.

    Attributes:
        data (DataPipelineConfig): Data configuration.
        dataset (DatasetPipelineConfig): Dataset configuration.
        data_loader (DataLoaderPipelineConfig): Dataloader configuration.
        early_stopping (EarlyStoppingPipelineConfig): Early stopping configuration.
        evaluation (EvaluationPipelineConfig): Evaluation configuration.
        logs (LogsPipelineConfig): Logs configuration.
        loss (LossPipelineConfig): Loss configuration.
        model (ModelPipelineConfig): Model configuration.
        optimizer (OptimizerPipelineConfig): Optimizer configuration.
        resources (ResourcesPipelineConfig): Resources configuration.
        seed (SeedPipelineConfig): Seed configuration.
        simulations (SimulationsPipelineConfig): Simulations configuration.
        training (TrainingPipelineConfig): Training configuration.
        validation (ValidationPipelineConfig): Validation configuration.
    """

    data: DataPipelineConfig
    dataset: DatasetPipelineConfig
    data_loader: DataLoaderPipelineConfig
    early_stopping: EarlyStoppingPipelineConfig
    evaluation: EvaluationPipelineConfig
    logs: LogsPipelineConfig
    loss: LossPipelineConfig
    model: ModelPipelineConfig
    optimizer: OptimizerPipelineConfig
    resources: ResourcesPipelineConfig
    seed: SeedPipelineConfig
    simulations: SimulationsPipelineConfig
    training: TrainingPipelineConfig
    validation: ValidationPipelineConfig
