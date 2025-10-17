from typing import List

from pydantic import BaseModel, confloat, conint, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from eviction_policy_api.const import (
    MAX_MC_DROPOUT_SAMPLES,
    MAX_ROLLOUT_HORIZON,
    MAX_TIME_STEP_INCREMENT,
    TIEBREAK_STRATEGIES,
)


class APIKwargs(BaseModel):
    """
    Kwargs for the API.

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
        exclude_keys (List[int]): List of keys that should not be evicted.
        prob_weight (float): Weight applied to probability in key scoring (in (0.0, 1.0]).
        conf_weight (float): Weight applied to confidence in key scoring (in (0.0, 1.0]).
        tiebreak_strategy (str): Strategy to resolve ties when multiple
                                 keys have equal scores.
        return_all_scores (bool): If True, API returns the score for every key.
        return_prob_conf (bool): If True, API returns the probability and confidence
                                 matrices used to calculate scores.
    """

    rollout_horizon: conint(gt=0, le=MAX_ROLLOUT_HORIZON)
    mc_dropout_samples: conint(gt=0, le=MAX_MC_DROPOUT_SAMPLES)
    confidence_level: confloat(gt=0.0, le=1.0)
    time_step_increment: confloat(gt=0, le=MAX_TIME_STEP_INCREMENT)
    num_evictions: conint(gt=0)
    exclude_keys: List[int]
    prob_weight: confloat(gt=0.0, le=1.0)
    conf_weight: confloat(gt=0.0, le=1.0)
    tiebreak_strategy: str
    return_all_scores: bool
    return_prob_conf: bool

    @model_validator(mode="after")
    def check_tiebreak_strategy(
        self: "APIKwargs",
    ) -> "APIKwargs":
        """
        Check whether tiebreak strategy is valid.

        This function validates the tiebreak strategy
        field of the API kwargs, ensuring its value
        is among the allowed strategies.

        Args:
            self ("APIKwargs"): Current model instance.

        Returns:
            "APIKwargs": Validated model instance.
        """
        assert_choice_field(
            self.tiebreak_strategy,
            TIEBREAK_STRATEGIES,
            "APIKwargs.tiebreak_strategy",
        )

        return self
