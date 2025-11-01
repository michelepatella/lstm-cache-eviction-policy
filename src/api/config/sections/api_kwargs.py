from typing import Optional

from pydantic import BaseModel, conint, confloat


class APIKwargInt(BaseModel):
    """
    Integer API kwarg configuration.

    Defines an API kwarg of integer type, including its
    default value and optional maximum allowed value.

    Attributes:
        default (int): Default integer value (> 0).
        max (Optional[int]): Maximum allowed integer value (> 0).
    """

    default: conint(gt=0)
    max: Optional[conint(gt=0)] = None


class APIKwargFloat(BaseModel):
    """
    Float API kwarg configuration.

    Defines an API kwarg of float type, including its
    default value and optional maximum allowed value.

    Attributes:
        default (float): Default float value (> 0.0).
        max (Optional[float]): Maximum allowed float value (> 0.0).
    """

    default: confloat(gt=0.0)
    max: Optional[confloat(gt=0.0)] = None


class APIKwargConfidenceLevel(BaseModel):
    """
    Confidence level API kwarg configuration.

    Defines the confidence level used for confidence
    intervals.

    Attributes:
        default (float): Default confidence level (0.0–1.0].
    """

    default: confloat(gt=0.0, le=1.0)


class APIKwargListInt(BaseModel):
    """
    List of integer API kwarg configuration.

    Defines an API kwarg containing a list of integer
    values.

    Attributes:
        default (list[int]): Default list of integers.
    """

    default: list[int]


class APIKwargBool(BaseModel):
    """
    Boolean API kwarg configuration.

    Defines an API kwarg of boolean type with a
    single default value.

    Attributes:
        default (bool): Default boolean value.
    """

    default: bool


class APIKwargs(BaseModel):
    """
    Complete API kwargs configuration.

    This class defines the full configuration for
    API kwargs.

    Attributes:
        rollout_horizon (APIKwargInt): Number of future steps to predict
                                       in the autoregressive rollout.
        mc_dropout_samples (APIKwargInt): Number of Monte Carlo Dropout samples
                                          for uncertainty estimation.
        confidence_level (APIKwargConfidenceLevel): Confidence level for
                                                    prediction intervals.
        time_step_increment (APIKwargFloat): Time step increment in hours
                                             for feature progression.
        num_evictions (APIKwargInt): Number of keys to evict per step.
        excluded_keys (APIKwargListInt): Keys that should not be evicted.
        prob_weight (APIKwargFloat): Weight applied to probability in key scoring.
        conf_weight (APIKwargFloat): Weight applied to confidence in key scoring.
        return_all_scores (APIKwargBool): Whether to return all scores.
        return_prob_conf (APIKwargBool): Whether to return probability and
                                         confidence matrices.
        return_api_kwargs (APIKwargBool): Whether to return all API kwargs in
                                          responses.
    """

    rollout_horizon: APIKwargInt
    mc_dropout_samples: APIKwargInt
    confidence_level: APIKwargConfidenceLevel
    time_step_increment: APIKwargFloat
    num_evictions: APIKwargInt
    excluded_keys: APIKwargListInt
    prob_weight: APIKwargFloat
    conf_weight: APIKwargFloat
    return_all_scores: APIKwargBool
    return_prob_conf: APIKwargBool
    return_api_kwargs: APIKwargBool
