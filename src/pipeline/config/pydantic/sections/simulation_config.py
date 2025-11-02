from pydantic import BaseModel, confloat, conint


class SimulationsMetricsMistakeRateConfig(BaseModel):
    """Configuration for mistake rate calculation.

    Attributes:
        window (int): Sliding window size used to compute the mistake rate (> 0).
    """

    window: conint(gt=0)


class SimulationsMetricsConfig(BaseModel):
    """Metrics configuration for the simulations.

    Attributes:
        mistake_rate (SimulationsMetricsMistakeRateConfig): Configuration for mistake rate metric.
    """

    mistake_rate: SimulationsMetricsMistakeRateConfig


class SimulationsCacheConfig(BaseModel):
    """Cache configuration for the simulations.

    Attributes:
        dimension (int): Number of cache slots (> 0).
        ttl (int): Time-to-Live for cached items (> 0).
    """

    dimension: conint(gt=0)
    ttl: conint(gt=0)


class SimulationsAPIKwargsConfig(BaseModel):
    """Configuration for API kwargs used in simulations.

    Attributes:
        rollout_horizon (int): Number of future time steps to roll out (> 0).
        mc_dropout_samples (int): Number of Monte Carlo Dropout samples (> 0).
        confidence_level (float): Confidence level used in uncertainty estimation
                                  (0 < x ≤ 1).
        time_step_increment (float): Increment for time step between
                                     predictions (> 0).
        num_evictions (int): Number of cache evictions to perform per step (≥ 0).
        excluded_keys (list[int]): Keys to exclude from eviction.
        prob_weight (float): Weight for probability-based scoring (0 ≤ x ≤ 1).
        conf_weight (float): Weight for confidence-based scoring (0 ≤ x ≤ 1).
        return_all_scores (bool): Whether to return all eviction scores.
        return_prob_conf (bool): Whether to return probability and confidence
                                 together.
        return_api_kwargs (bool): Whether to return the API kwargs in the
                                  response.
    """

    rollout_horizon: conint(gt=0)
    mc_dropout_samples: conint(gt=0)
    confidence_level: confloat(gt=0, le=1)
    time_step_increment: confloat(gt=0)
    num_evictions: conint(ge=0)
    excluded_keys: list[int]
    prob_weight: confloat(ge=0, le=1)
    conf_weight: confloat(ge=0, le=1)
    return_all_scores: bool
    return_prob_conf: bool
    return_api_kwargs: bool


class SimulationsConfig(BaseModel):
    """Simulations configuration.

    Attributes:
        cache (SimulationsCacheConfig): Cache configuration.
        metrics (SimulationsMetricsConfig): Simulations metrics configuration.
        api_kwargs (SimulationsAPIKwargsConfig): API kwargs configuration.
    """

    cache: SimulationsCacheConfig
    metrics: SimulationsMetricsConfig
    api_kwargs: SimulationsAPIKwargsConfig
