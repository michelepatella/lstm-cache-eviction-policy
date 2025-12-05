"""simulations_config.py

Configuration section for the simulation environment and parameters.

This module defines advanced settings for running predictive simulations,
including specific API arguments and configuration for the underlying caching
mechanisms.

Classes:
    SimulationsApiKwargsConfig(BaseModel):
        Configuration for API-specific arguments (e.g., weights, levels).
    SimulationsCachesConfig(BaseModel):
        Configuration for general cache settings (e.g., size, TTL).
    SimulationsConfig(BaseModel):
        Aggregates all configuration settings for simulations.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class SimulationsApiKwargsConfig(BaseModel):
    """Configuration for API-specific arguments.

    Attributes:
        conf_weight (float): Weight applied to the confidence score
                             in decision-making (in (0,1]).
        conf_level (float): Minimum confidence level required
                            for certain decisions (in (0,1]).
        excluded_keys (list[int]): List of keys to exclude from cache operations.
        mc_dropout_samples (int): Number of Monte Carlo Dropout samples
                                  to use for uncertainty estimation (>= 1).
        num_evictions (int): Number of items to evict simultaneously (>= 1).
        prob_weight (float): Weight applied to the prediction probability
                             score (in (0,1]).
        return_all_scores (bool): Flag to return all scores from the API.
        return_api_kwargs (bool): Flag to return API arguments alongside results.
        return_prob_conf (bool): Flag to return probability and confidence scores.
        rollout_horizon (int): Time horizon for autoregressive rollout
                               prediction (>= 1).
        time_step_increment (float): Time increment used in autoregressive
                                     rollout (> 0).
    """

    conf_weight: Annotated[float, Field(gt=0, le=1)]
    conf_level: Annotated[float, Field(gt=0, le=1)]
    excluded_keys: list[int]
    mc_dropout_samples: Annotated[int, Field(ge=1)]
    num_evictions: Annotated[int, Field(ge=1)]
    prob_weight: Annotated[float, Field(gt=0, le=1)]
    return_all_scores: bool
    return_api_kwargs: bool
    return_prob_conf: bool
    rollout_horizon: Annotated[int, Field(ge=1)]
    time_step_increment: Annotated[float, Field(gt=0)]


class SimulationsCachesConfig(BaseModel):
    """Configuration for cache settings.

    Attributes:
        dimension (int): The size (number of keys) of the cache (>= 1).
        ttl (int): Time-To-Live (in seconds) for cache entries (>= 0).
    """

    dimension: Annotated[int, Field(ge=1)]
    ttl: Annotated[int, Field(ge=0)]


class SimulationsConfig(BaseModel):
    """Simulations configuration.

    Attributes:
        api_kwargs (SimulationsApiKwargsConfig): Keyword arguments passed to the
                                                 internal simulation API calls.
        caches (SimulationsCachesConfig): Configuration for the caching mechanism used
                                          within the simulation environment.
    """

    api_kwargs: SimulationsApiKwargsConfig
    caches: SimulationsCachesConfig
