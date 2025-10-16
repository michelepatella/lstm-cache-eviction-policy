from pydantic import BaseModel, conint


class MistakeRateConfig(BaseModel):
    """
    Configuration for mistake rate calculation.

    Attributes:
        window (int): Sliding window size used to compute the
                      mistake rate (> 0).
    """

    window: conint(gt=0)


class MetricsConfig(BaseModel):
    """
    Metrics configuration for the simulations.

    Attributes:
        mistake_rate (MistakeRateConfig): Configuration for mistake
                                          rate metric.
    """

    mistake_rate: MistakeRateConfig


class CacheConfig(BaseModel):
    """
    Cache configuration for the simulations.

    Attributes:
        dimension (int): Number of cache slots (> 0).
        ttl (int): Time-to-Live for cached items (> 0).
    """

    dimension: conint(gt=0)
    ttl: conint(gt=0)


class SimulationsConfig(BaseModel):
    """
    Simulations configuration.

    Attributes:
        cache (CacheConfig): Cache configuration.
        metrics (MetricsConfig): Simulations metrics configuration.
    """

    cache: CacheConfig
    metrics: MetricsConfig
