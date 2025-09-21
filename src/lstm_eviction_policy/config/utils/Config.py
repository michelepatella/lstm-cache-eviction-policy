from pydantic import BaseModel

from lstm_eviction_policy.config.utils.DataConfig import DataConfig
from lstm_eviction_policy.config.utils.EvaluationConfig import EvaluationConfig
from lstm_eviction_policy.config.utils.InferenceConfig import InferenceConfig
from lstm_eviction_policy.config.utils.ModelConfig import ModelConfig
from lstm_eviction_policy.config.utils.SimulationConfig import SimulationConfig
from lstm_eviction_policy.config.utils.TestingConfig import TestingConfig
from lstm_eviction_policy.config.utils.TrainingConfig import TrainingConfig
from lstm_eviction_policy.config.utils.ValidationConfig import ValidationConfig


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
