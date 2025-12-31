"""simulations_pipeline_config.py

Configuration section for the simulation environment and parameters.

This module defines advanced settings for running predictive simulations,
including specific API arguments and configuration for the underlying caching
mechanisms.

Classes:
    SimulationsApiKwargsPipelineConfig(BaseModel):
        Configuration for API-specific arguments (e.g., weights, levels).
    SimulationsCachesPipelineConfig(BaseModel):
        Configuration for general cache settings (e.g., size, TTL).
    SimulationsPipelineConfig(BaseModel):
        Aggregates all configuration settings for simulations.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class SimulationsApiKwargsPipelineConfig(BaseModel):
    """Configuration for API-specific arguments.

    Attributes:
        conf_weight (float): Weight applied to the confidence score
                             in decision-making (in (0.0, 5.0]).
        conf_level (float): Minimum confidence level required
                            for certain decisions (in (0.0, 1.0]).
        excluded_keys (list[int]): List of keys to exclude from cache operations.
        mc_dropout_samples (int): Number of Monte Carlo Dropout samples
                                  to use for uncertainty estimation (>= 1).
        num_evictions (int): Number of items to evict simultaneously (>= 1).
        rollout_horizon (int): Time horizon for autoregressive rollout
                               prediction (>= 1).
        time_step_increment (float): Time increment used in autoregressive
                                     rollout (> 0).
    """

    conf_weight: Annotated[float, Field(gt=0.0, le=5.0)]
    conf_level: Annotated[float, Field(gt=0, le=1)]
    excluded_keys: list[int]
    mc_dropout_samples: Annotated[int, Field(ge=1)]
    num_evictions: Annotated[int, Field(ge=1)]
    rollout_horizon: Annotated[int, Field(ge=1)]
    time_step_increment: Annotated[float, Field(gt=0)]


class SimulationsCachesPipelineConfig(BaseModel):
    """Configuration for cache settings.

    Attributes:
        dimension (int): The size (number of keys) of the cache (>= 1).
        ttl (int): Time-To-Live (in seconds) for cache entries (>= 0).
    """

    dimension: Annotated[int, Field(ge=1)]
    ttl: Annotated[int, Field(ge=0)]


class SimulationsPipelineConfig(BaseModel):
    """Simulations configuration.

    Attributes:
        api_kwargs (SimulationsApiKwargsPipelineConfig): Keyword arguments passed
                                                         to the internal simulation
                                                         API calls.
        caches (SimulationsCachesPipelineConfig): Configuration for the caching
                                                  mechanism used within the simulation
                                                  environment.
    """

    api_kwargs: SimulationsApiKwargsPipelineConfig
    caches: SimulationsCachesPipelineConfig
