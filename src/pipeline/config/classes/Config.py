from typing import Dict, List, TypeAlias, Union

from pydantic import BaseModel

from pipeline.config.classes.DataConfig import DataConfig
from pipeline.config.classes.DatasetConfig import DatasetConfig
from pipeline.config.classes.HardwareConfig import HardwareConfig
from pipeline.config.classes.ModelConfig import ModelConfig
from pipeline.config.classes.SimulationConfig import SimulationConfig
from pipeline.config.classes.TestingConfig import TestingConfig
from pipeline.config.classes.TrainingConfig import TrainingConfig
from pipeline.config.classes.ValidationConfig import ValidationConfig

# Definition of configuration value
# and dictionary types
ConfigValue: TypeAlias = Union[
    int,
    float,
    str,
    bool,
    List[Union[int, float]],
    Dict[str, "ConfigValue"],
]
ConfigDict: TypeAlias = Dict[str, ConfigValue]


class Config(BaseModel):
    """
    Represents the full configuration of the system.

    This class aggregates all configuration sections used
    across the system, providing structured access
    to hardware, data, dataset, model, simulation, training,
    testing, and validation configurations.

    Attributes:
        hardware (HardwareConfig): Hardware configuration.
        data (DataConfig): Data configuration.
        dataset (DatasetConfig): Dataset configuration.
        model (ModelConfig): Model configuration.
        validation (ValidationConfig): Validation configuration.
        training (TrainingConfig): Training configuration.
        testing (TestingConfig): Testing configuration.
        simulation (SimulationConfig): Simulation configuration.
    """

    hardware: HardwareConfig
    data: DataConfig
    dataset: DatasetConfig
    model: ModelConfig
    validation: ValidationConfig
    training: TrainingConfig
    testing: TestingConfig
    simulation: SimulationConfig
