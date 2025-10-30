from pydantic import BaseModel, confloat, conint

from api.const import (
    API_KWARGS_MC_DROPOUT_SAMPLES_MAX,
    API_KWARGS_ROLLOUT_HORIZON_MAX,
    API_KWARGS_TIME_STEP_INCREMENT_MAX,
)


class APIKwargs(BaseModel):
    """Kwargs for the API.

    This class encapsulates all the API kwargs, checking
    their validity.

    Attributes:
        rollout_horizon (int): Number of future steps to predict in the
                               autoregressive rollout (in (0, MAX_ROLLOUT_HORIZON]).
        mc_dropout_samples (int): Number of Monte Carlo dropout samples
                                  for uncertainty estimation (in (0, MAX_MC_DROPOUT_SAMPLES]).
        confidence_level (float): Confidence level for prediction intervals
                                  (in (0.0, 1.0]).
        time_step_increment (float): Time step increment in hours for feature
                                     progression (in (0, MAX_TIME_STEP_INCREMENT]).
        num_evictions (int): Number of keys to evict in the current step (> 0).
        excluded_keys (List[int]): List of keys that should not be evicted.
        prob_weight (float): Weight applied to probability in key scoring (in (0.0, 1.0]).
        conf_weight (float): Weight applied to confidence in key scoring (in (0.0, 1.0]).
        return_all_scores (bool): If True, API returns the score for every key.
        return_prob_conf (bool): If True, API returns the probability and confidence
                                 matrices used to calculate scores.
        return_api_kwargs (bool): If True, API returns all API kwargs.
    """

    rollout_horizon: conint(gt=0, le=API_KWARGS_ROLLOUT_HORIZON_MAX)
    mc_dropout_samples: conint(gt=0, le=API_KWARGS_MC_DROPOUT_SAMPLES_MAX)
    confidence_level: confloat(gt=0.0, le=1.0)
    time_step_increment: confloat(gt=0, le=API_KWARGS_TIME_STEP_INCREMENT_MAX)
    num_evictions: conint(gt=0)
    excluded_keys: list[int]
    prob_weight: confloat(gt=0.0, le=1.0)
    conf_weight: confloat(gt=0.0, le=1.0)
    return_all_scores: bool
    return_prob_conf: bool
    return_api_kwargs: bool
