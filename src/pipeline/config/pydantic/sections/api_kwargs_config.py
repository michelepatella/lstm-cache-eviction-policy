"""api_kwargs_config.py

Configuration section for parameters passed to the API.

This module defines the structure for advanced settings used for API,
including confidence levels, weights, rollout horizon, and control flags.

Classes:
    ApiKwargsConfig(BaseModel):
        Configuration for API-specific arguments.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ApiKwargsConfig(BaseModel):
    """Configuration for API-specific arguments.

    Attributes:
        conf_weight (float): Weight applied to the confidence score
                                in decision-making (> 0, <= 1).
        confidence_level (float): Minimum confidence level required
                                     for certain decisions (> 0, <= 1).
        excluded_keys (list[int]): List of keys to exclude from cache operations.
        mc_dropout_samples (int): Number of Monte Carlo Dropout samples
                                     to use for uncertainty estimation (>= 1).
        num_evictions (int): Number of items to evict simultaneously (>= 1).
        prob_weight (float): Weight applied to the prediction probability
                                score (> 0, <= 1).
        return_all_scores (bool): Flag to return all scores from the API.
        return_api_kwargs (bool): Flag to return API arguments alongside results.
        return_prob_conf (bool): Flag to return probability and confidence scores.
        rollout_horizon (int): Time horizon for autoregressive rollout
                                  prediction (>= 1).
        time_step_increment (float): Time increment used in autoregressive
                                        rollout (> 0).
    """

    conf_weight: Annotated[float, Field(gt=0, le=1)]
    confidence_level: Annotated[float, Field(gt=0, le=1)]
    excluded_keys: list[int]
    mc_dropout_samples: Annotated[int, Field(ge=1)]
    num_evictions: Annotated[int, Field(ge=1)]
    prob_weight: Annotated[float, Field(gt=0, le=1)]
    return_all_scores: bool
    return_api_kwargs: bool
    return_prob_conf: bool
    rollout_horizon: Annotated[int, Field(ge=1)]
    time_step_increment: Annotated[float, Field(gt=0)]
