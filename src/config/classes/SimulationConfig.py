from pydantic import BaseModel, confloat, conint


# Simulation — Cache — General
class CacheGeneralConfig(BaseModel):
    dimension: conint(gt=0)  # type: ignore[valid-type]
    ttl: conint(gt=0)  # type: ignore[valid-type]


# Simulation — Cache — LSTM
class LSTMCachePredictionConfig(BaseModel):
    interval: conint(gt=0)  # type: ignore[valid-type]


class LSTMCacheConfig(BaseModel):
    prediction: LSTMCachePredictionConfig
    threshold: confloat(ge=0, le=1)  # type: ignore[valid-type]


# Simulation — Evaluation — Mistake Rate
class MistakeRateConfig(BaseModel):
    window: conint(gt=0)  # type: ignore[valid-type]


class EvaluationConfig(BaseModel):
    mistake_rate: MistakeRateConfig


class SimulationConfig(BaseModel):
    """
    Class representing the simulation configuration
    settings including cache settings — general and LSTM-related.
    """

    general: CacheGeneralConfig
    lstm: LSTMCacheConfig
    evaluation: EvaluationConfig
