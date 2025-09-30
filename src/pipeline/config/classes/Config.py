from typing import Dict, List, TypeAlias, Union

from pydantic import BaseModel

from pipeline.config.classes.DataConfig import (
    DataConfig,
)
from pipeline.config.classes.EvaluationConfig import (
    EvaluationConfig,
)
from pipeline.config.classes.HardwareConfig import HardwareConfig
from pipeline.config.classes.InferenceConfig import (
    InferenceConfig,
)
from pipeline.config.classes.ModelConfig import (
    ModelConfig,
)
from pipeline.config.classes.SimulationConfig import (
    SimulationConfig,
)
from pipeline.config.classes.TestingConfig import (
    TestingConfig,
)
from pipeline.config.classes.TrainingConfig import (
    TrainingConfig,
)
from pipeline.config.classes.ValidationConfig import (
    ValidationConfig,
)

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
