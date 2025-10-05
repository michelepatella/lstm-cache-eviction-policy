from typing import Dict, List, TypeAlias, Union

from pydantic import BaseModel

from config.classes.DataConfig import DataConfig
from config.classes.EvaluationConfig import EvaluationConfig
from config.classes.HardwareConfig import HardwareConfig
from config.classes.InferenceConfig import InferenceConfig
from config.classes.ModelConfig import ModelConfig
from config.classes.SimulationConfig import SimulationConfig
from config.classes.TestingConfig import TestingConfig
from config.classes.TrainingConfig import TrainingConfig
from config.classes.ValidationConfig import ValidationConfig

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
    Class representing the whole
    configuration settings.
    """

    hardware: HardwareConfig
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    training: TrainingConfig
    testing: TestingConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    simulation: SimulationConfig
