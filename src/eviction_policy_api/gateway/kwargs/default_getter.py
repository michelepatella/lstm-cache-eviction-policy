from typing import Dict, List

from box import Box

from eviction_policy_api.const import (
    CONF_WEIGHT_API_KWARG_NAME,
    CONFIDENCE_LEVEL_API_KWARG_NAME,
    EXCLUDE_KEYS_API_KWARG_NAME,
    MC_DROPOUT_SAMPLES_API_KWARG_NAME,
    NUM_EVICTIONS_API_KWARG_NAME,
    PROB_WEIGHT_API_KWARG_NAME,
    RETURN_ALL_SCORES_API_KWARG_NAME,
    RETURN_PROB_CONF_API_KWARG_NAME,
    ROLLOUT_HORIZON_API_KWARG_NAME,
    TIEBREAK_STRATEGY_API_KWARG_NAME,
    TIME_STEP_INCREMENT_API_KWARG_NAME,
)


def get_default_kwargs(
    api_config: Box,
) -> Dict[str, int | float | List[int] | str | bool]:
    """
    Retrieve the default API kwarg values.

    This function extracts all the default API kwarg values
    from the configuration object, returning them.

    Args:
        api_config (Box): API configuration object.

    Returns:
        Dict[str, int | float | List[int] | str | bool]: Dictionary
            mapping each API kwarg name to its default value.
    """
    default_kwargs: Dict[str, int | float | List[int] | str | bool] = {
        ROLLOUT_HORIZON_API_KWARG_NAME: api_config.kwargs.rollout_horizon,
        MC_DROPOUT_SAMPLES_API_KWARG_NAME: api_config.kwargs.mc_dropout_samples,
        CONFIDENCE_LEVEL_API_KWARG_NAME: api_config.kwargs.confidence_level,
        TIME_STEP_INCREMENT_API_KWARG_NAME: api_config.kwargs.time_step_increment,
        NUM_EVICTIONS_API_KWARG_NAME: api_config.kwargs.num_evictions,
        EXCLUDE_KEYS_API_KWARG_NAME: api_config.kwargs.exclude_keys,
        PROB_WEIGHT_API_KWARG_NAME: api_config.kwargs.prob_weight,
        CONF_WEIGHT_API_KWARG_NAME: api_config.kwargs.conf_weight,
        TIEBREAK_STRATEGY_API_KWARG_NAME: api_config.kwargs.tiebreak_strategy,
        RETURN_ALL_SCORES_API_KWARG_NAME: api_config.kwargs.return_all_scores,
        RETURN_PROB_CONF_API_KWARG_NAME: api_config.kwargs.return_prob_conf,
    }

    return default_kwargs
