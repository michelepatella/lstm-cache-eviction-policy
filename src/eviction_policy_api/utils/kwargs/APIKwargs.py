from typing import List

from pydantic import BaseModel, confloat, conint


class APIKwargs(BaseModel):
    """
    Kwargs for the API.

    This class encapsulates all the API kwargs, checking
    their validity.

    Attributes:
        rollout_horizon (int): Number of future steps to predict in the
                               autoregressive rollout (> 0).
        mc_dropout_samples (int): Number of Monte Carlo dropout samples
                                  for uncertainty estimation (> 0).
        confidence_level (float): Confidence level for prediction intervals.
                                  (in (0.0, 1.0])
        time_step_increment (float): Time step increment in hours for feature
                                     progression (> 0).
        num_evictions (int): Number of keys to evict in the current step (> 0).
        exclude_keys (List[int]): List of keys that should not be evicted.
        prob_weight (float): Weight applied to probability in key scoring.
                             (in [0.0, 1.0]).
        conf_weight (float): Weight applied to confidence in key scoring.
                             (in [0.0, 1.0]).
        tiebreak_strategy (str): Strategy to resolve ties when multiple
                                 keys have equal scores.
        return_all_scores (bool): If True, API returns the score for every key.
        return_prob_conf (bool): If True, API returns the probability and confidence matrices
                                 used to calculate scores.
    """

    rollout_horizon: conint(gt=0)
    mc_dropout_samples: conint(gt=0)
    confidence_level: confloat(gt=0.0, le=1.0)
    time_step_increment: confloat(gt=0)
    num_evictions: conint(gt=0)
    exclude_keys: List[int]
    prob_weight: confloat(ge=0.0, le=1.0)
    conf_weight: confloat(ge=0.0, le=1.0)
    tiebreak_strategy: str
    return_all_scores: bool
    return_prob_conf: bool
