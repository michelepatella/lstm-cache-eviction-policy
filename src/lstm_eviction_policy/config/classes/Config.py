from typing import Dict, List, TypeAlias, Union

from pydantic import BaseModel

from lstm_eviction_policy.config.classes.DataConfig import (
    DataConfig,
)
from lstm_eviction_policy.config.classes.EvaluationConfig import (
    EvaluationConfig,
)
from lstm_eviction_policy.config.classes.InferenceConfig import (
    InferenceConfig,
)
from lstm_eviction_policy.config.classes.ModelConfig import (
    ModelConfig,
)
from lstm_eviction_policy.config.classes.SimulationConfig import (
    SimulationConfig,
)
from lstm_eviction_policy.config.classes.TestingConfig import (
    TestingConfig,
)
from lstm_eviction_policy.config.classes.TrainingConfig import (
    TrainingConfig,
)
from lstm_eviction_policy.config.classes.ValidationConfig import (
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

    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    training: TrainingConfig
    testing: TestingConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    simulation: SimulationConfig
