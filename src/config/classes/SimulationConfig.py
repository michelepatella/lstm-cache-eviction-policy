from pydantic import BaseModel, confloat, conint


# Simulation — Cache — General
class CacheGeneralConfig(BaseModel):
    dimension: conint(gt=0)
    ttl: conint(gt=0)


# Simulation — Cache — LSTM — Confidence Intervals
class ConfidenceIntervalsConfig(BaseModel):
    level: confloat(ge=0, le=1)


class MCDropoutConfig(BaseModel):
    samples: conint(gt=0)


# Simulation — Cache — LSTM — Main Config
class LSTMCacheConfig(BaseModel):
    prediction_interval: conint(gt=0)
    threshold: confloat(ge=0, le=1)
    confidence_intervals: ConfidenceIntervalsConfig
    mc_dropout: MCDropoutConfig


# Simulation — Metrics — Mistake Rate
class MistakeRateConfig(BaseModel):
    window: conint(gt=0)


class MetricsConfig(BaseModel):
    mistake_rate: MistakeRateConfig


# Simulation — Main Config
class CacheConfig(BaseModel):
    general: CacheGeneralConfig
    lstm: LSTMCacheConfig


class SimulationConfig(BaseModel):
    cache: CacheConfig
    metrics: MetricsConfig
