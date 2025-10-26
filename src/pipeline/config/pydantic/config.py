from pydantic import BaseModel

from pipeline.config.pydantic.sections.data_config import DataConfig
from pipeline.config.pydantic.sections.dataset_config import DatasetConfig
from pipeline.config.pydantic.sections.hardware_config import HardwareConfig
from pipeline.config.pydantic.sections.model_config import ModelConfig
from pipeline.config.pydantic.sections.simulation_config import (
    SimulationsConfig,
)
from pipeline.config.pydantic.sections.testing_config import TestingConfig
from pipeline.config.pydantic.sections.training_config import TrainingConfig
from pipeline.config.pydantic.sections.validation_config import (
    ValidationConfig,
)


class Config(BaseModel):
    """Represents the full configuration of the system.

    This class aggregates all configuration sections used across the system,
    providing structured access to hardware, data, dataset, model, validation,
    training, testing, and simulations configurations.

    Attributes:
        hardware (HardwareConfig): Hardware configuration.
        data (DataConfig): Data configuration.
        dataset (DatasetConfig): Dataset configuration.
        model (ModelConfig): Model configuration.
        validation (ValidationConfig): Validation configuration.
        training (TrainingConfig): Training configuration.
        testing (TestingConfig): Testing configuration.
        simulations (SimulationsConfig): Simulations configuration.
    """

    hardware: HardwareConfig
    data: DataConfig
    dataset: DatasetConfig
    model: ModelConfig
    validation: ValidationConfig
    training: TrainingConfig
    testing: TestingConfig
    simulations: SimulationsConfig
