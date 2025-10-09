from pydantic import BaseModel, confloat, conint


# Simulation — Metrics — Mistake Rate
class MistakeRateConfig(BaseModel):
    window: conint(gt=0)


class MetricsConfig(BaseModel):
    mistake_rate: MistakeRateConfig


# Simulation —  Cache
class CacheConfig(BaseModel):
    dimension: conint(gt=0)
    ttl: conint(gt=0)


class SimulationConfig(BaseModel):
    cache: CacheConfig
    metrics: MetricsConfig
