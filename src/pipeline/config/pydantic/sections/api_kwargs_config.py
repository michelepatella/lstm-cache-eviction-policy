"""api_kwargs_config.py

Configuration section for parameters passed to the API.

This module defines the structure for advanced settings used for API,
including confidence levels, weights, rollout horizon, and control flags.

Classes:
    ApiKwargsConfig: Configuration for API-specific arguments.
"""

from pydantic import BaseModel, confloat, conint


class ApiKwargsConfig(BaseModel):
    """Configuration for API-specific arguments.

    Attributes:
        conf_weight (confloat): Weight applied to the confidence score
                                in decision-making (> 0, <= 1).
        confidence_level (confloat): Minimum confidence level required
                                     for certain decisions (> 0, <= 1).
        excluded_keys (list[int]): List of keys to exclude from cache operations.
        mc_dropout_samples (conint): Number of Monte Carlo Dropout samples
                                     to use for uncertainty estimation (>= 1).
        num_evictions (conint): Number of items to evict simultaneously (>= 1).
        prob_weight (confloat): Weight applied to the prediction probability
                                score (> 0, <= 1).
        return_all_scores (bool): Flag to return all scores from the API.
        return_api_kwargs (bool): Flag to return API arguments alongside results.
        return_prob_conf (bool): Flag to return probability and confidence scores.
        rollout_horizon (conint): Time horizon for autoregressive rollout
                                  prediction (>= 1).
        time_step_increment (confloat): Time increment used in autoregressive
                                        rollout (> 0).
    """

    conf_weight: confloat(gt=0, le=1)
    confidence_level: confloat(gt=0, le=1)
    excluded_keys: list[int]
    mc_dropout_samples: conint(ge=1)
    num_evictions: conint(ge=1)
    prob_weight: confloat(gt=0, le=1)
    return_all_scores: bool
    return_api_kwargs: bool
    return_prob_conf: bool
    rollout_horizon: conint(ge=1)
    time_step_increment: confloat(gt=0)
