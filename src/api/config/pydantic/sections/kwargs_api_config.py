"""api_kwargs_api_config.py

Module dedicated to defining Pydantic schemas for the complete set
of keyword arguments (kwargs) for the API.

These schemas enforce validation rules for various data types—including
integers, floats, boolean flags, and lists—to ensure the functional
and statistical integrity of the API calls. This includes specifying
default values, boundary constraints, and required data types for
parameters controlling autoregression, uncertainty quantification
(MC Dropout), and cache eviction logic.

Classes:
    KwargIntAPIConfig(BaseModel):
        Configuration for integer kwargs with boundary checks.
    KwargFloatAPIConfig(BaseModel):
        Configuration for float kwargs with boundary checks.
    KwargConfidenceLevelAPIConfig(BaseModel):
        Configuration for float kwargs constrained to be a probability/level
        (0.0 < x <= 1.0).
    KwargListIntAPIConfig(BaseModel):
        Configuration for list-of-integer kwargs.
    KwargBoolAPIConfig(BaseModel):
        Configuration for boolean kwargs.
    KwargsAPIConfig(BaseModel):
        The complete, top-level schema defining all supported API kwargs.
"""

from pydantic import BaseModel, confloat, conint


class KwargIntAPIConfig(BaseModel):
    """Integer API kwarg configuration.

    Attributes:
        default (int): Default integer value (> 0).
        max (Optional[int]): Maximum allowed integer value (> 0).
    """

    default: conint(gt=0)
    max: conint(gt=0) | None = None


class KwargFloatAPIConfig(BaseModel):
    """Float API kwarg configuration.

    Defines an API kwarg of float type, including its
    default value and optional maximum allowed value.

    Attributes:
        default (float): Default float value (> 0.0).
        max (Optional[float]): Maximum allowed float value (> 0.0).
    """

    default: confloat(gt=0.0)
    max: confloat(gt=0.0) | None = None


class KwargConfidenceLevelAPIConfig(BaseModel):
    """Confidence level API kwarg configuration.

    Defines the confidence level used for confidence
    intervals.

    Attributes:
        default (float): Default confidence level (0.0–1.0].
    """

    default: confloat(gt=0.0, le=1.0)


class KwargListIntAPIConfig(BaseModel):
    """List of integer API kwarg configuration.

    Defines an API kwarg containing a list of integer
    values.

    Attributes:
        default (list[int]): Default list of integers.
    """

    default: list[int]


class KwargBoolAPIConfig(BaseModel):
    """Boolean API kwarg configuration.

    Defines an API kwarg of boolean type with a
    single default value.

    Attributes:
        default (bool): Default boolean value.
    """

    default: bool


class KwargsAPIConfig(BaseModel):
    """Complete API kwargs configuration.

    This class defines the full configuration for
    API kwargs.

    Attributes:
        rollout_horizon (KwargIntAPIConfig): Number of future steps to predict
                                       in the autoregressive rollout.
        mc_dropout_samples (KwargIntAPIConfig): Number of Monte Carlo Dropout samples
                                          for uncertainty estimation.
        confidence_level (KwargConfidenceLevelAPIConfig): Confidence level for
                                                    prediction intervals.
        time_step_increment (KwargFloatAPIConfig): Time step increment in hours
                                             for feature progression.
        num_evictions (KwargIntAPIConfig): Number of keys to evict per step.
        excluded_keys (KwargListIntAPIConfig): Keys that should not be evicted.
        prob_weight (KwargFloatAPIConfig): Weight applied to probability in key scoring.
        conf_weight (KwargFloatAPIConfig): Weight applied to confidence in key scoring.
        return_all_scores (KwargBoolAPIConfig): Whether to return all scores.
        return_prob_conf (KwargBoolAPIConfig): Whether to return probability and
                                         confidence matrices.
        return_api_kwargs (KwargBoolAPIConfig): Whether to return all API kwargs in
                                          responses.
    """

    rollout_horizon: KwargIntAPIConfig
    mc_dropout_samples: KwargIntAPIConfig
    confidence_level: KwargConfidenceLevelAPIConfig
    time_step_increment: KwargFloatAPIConfig
    num_evictions: KwargIntAPIConfig
    excluded_keys: KwargListIntAPIConfig
    unbiased_variance: KwargBoolAPIConfig
    prob_weight: KwargFloatAPIConfig
    conf_weight: KwargFloatAPIConfig
    return_all_scores: KwargBoolAPIConfig
    return_prob_conf: KwargBoolAPIConfig
    return_api_kwargs: KwargBoolAPIConfig
