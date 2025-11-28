"""api_kwargs_api_config.py

Module dedicated to defining Pydantic schemas for the complete set
of keyword arguments (kwargs) for the API.

These schemas enforce validation rules for various data types, including
integers, floats, boolean flags, and lists, to ensure the functional
and statistical integrity of the API calls. This includes specifying
default values, boundary constraints, and required data types for
parameters controlling autoregression, uncertainty quantification
(MC Dropout), and cache eviction logic.

Classes:
    KwargIntAPIConfig(BaseModel):
        Configuration for integer kwargs with boundary checks.
    KwargFloatAPIConfig(BaseModel):
        Configuration for float kwargs with boundary checks.
    KwargConfLevelAPIConfig(BaseModel):
        Configuration for float kwargs constrained to be a probability/level.
    KwargListIntAPIConfig(BaseModel):
        Configuration for list-of-integer kwargs.
    KwargBoolAPIConfig(BaseModel):
        Configuration for boolean kwargs.
    KwargsAPIConfig(BaseModel):
        The complete, top-level schema defining all supported API kwargs.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class KwargIntAPIConfig(BaseModel):
    """Integer API kwarg configuration.

    Attributes:
        value (int): Integer value (> 0).
        max (Optional[int]): Maximum allowed integer value (> 0).
    """

    value: Annotated[int, Field(gt=0)]
    max: Annotated[int, Field(gt=0)] | None = None


class KwargFloatAPIConfig(BaseModel):
    """Float API kwarg configuration.

    Attributes:
        value (float): Float value (> 0.0).
        max (Optional[float]): Maximum allowed float value (> 0.0).
    """

    value: Annotated[float, Field(gt=0.0)]
    max: Annotated[float, Field(gt=0.0)] | None = None


class KwargConfLevelAPIConfig(BaseModel):
    """Confidence level API kwarg configuration.

    Attributes:
        value (float):  Confidence level value (0.0–1.0].
    """

    value: Annotated[float, Field(gt=0.0, le=1.0)]


class KwargListIntAPIConfig(BaseModel):
    """List of integer API kwarg configuration.

    Attributes:
        value (list[int]): List of integer values.
    """

    value: list[int]


class KwargBoolAPIConfig(BaseModel):
    """Boolean API kwarg configuration.

    Attributes:
        value (bool): Boolean value.
    """

    value: bool


class KwargsAPIConfig(BaseModel):
    """Complete API kwargs configuration.

    Attributes:
        rollout_horizon (KwargIntAPIConfig): Number of future steps to predict
                                             in the autoregressive rollout.
        mc_dropout_samples (KwargIntAPIConfig): Number of Monte Carlo Dropout samples
                                                for uncertainty estimation.
        conf_level (KwargConfLevelAPIConfig): Confidence level for
                                                    prediction intervals.
        time_step_increment (KwargFloatAPIConfig): Time step increment in hours
                                                   for feature progression.
        num_evictions (KwargIntAPIConfig): Number of keys to evict per step.
        excluded_keys (KwargListIntAPIConfig): Keys that should not be evicted.
        prob_weight (KwargFloatAPIConfig): Weight applied to probability in key scoring.
        conf_weight (KwargFloatAPIConfig): Weight applied to confidence in key scoring.
    """

    rollout_horizon: KwargIntAPIConfig
    mc_dropout_samples: KwargIntAPIConfig
    conf_level: KwargConfLevelAPIConfig
    time_step_increment: KwargFloatAPIConfig
    num_evictions: KwargIntAPIConfig
    excluded_keys: KwargListIntAPIConfig
    unbiased_variance: KwargBoolAPIConfig
    prob_weight: KwargFloatAPIConfig
    conf_weight: KwargFloatAPIConfig
