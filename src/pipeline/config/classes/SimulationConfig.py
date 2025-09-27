from pydantic import BaseModel, confloat, conint


# Simulation — Cache — General
class CacheGeneralConfig(BaseModel):
    dimension: conint(gt=0)
    ttl: conint(gt=0)


# Simulation — Cache — LSTM
class LSTMCachePredictionConfig(BaseModel):
    interval: conint(gt=0)


class LSTMCacheConfig(BaseModel):
    prediction: LSTMCachePredictionConfig
    threshold: confloat(ge=0, le=1)


class SimulationConfig(BaseModel):
    """
    Class representing the simulation configuration
    settings including cache settings — general and LSTM-related.
    """

    general: CacheGeneralConfig
    lstm: LSTMCacheConfig
