from box import Box
from fastapi import HTTPException, status

from api.const import (
    CONF_WEIGHT_API_KWARG_NAME,
    CONFIDENCE_LEVEL_API_KWARG_NAME,
    EXCLUDED_KEYS_API_KWARG_NAME,
    MC_DROPOUT_SAMPLES_API_KWARG_NAME,
    NUM_EVICTIONS_API_KWARG_NAME,
    PROB_WEIGHT_API_KWARG_NAME,
    RETURN_ALL_SCORES_API_KWARG_NAME,
    RETURN_PROB_CONF_API_KWARG_NAME,
    ROLLOUT_HORIZON_API_KWARG_NAME,
    TIME_STEP_INCREMENT_API_KWARG_NAME,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def get_default_api_kwargs(
    api_config: Box,
) -> dict[str, int | float | list[int] | str | bool]:
    """Retrieve the default API kwarg values.

    This function extracts all the default API kwarg values
    from the configuration object, returning them.

    Args:
        api_config (Box): API configuration object.

    Returns:
        dict[str, int | float | list[int] | str | bool]:
            Dictionary mapping each API kwarg name
            to its default value.

    Raises:
        HTTPException: If retrieving default kwargs fails:
            * If the configuration structure is malformed (AttributeError).
            * If any expected key is missing (KeyError).
            * If value types in configuration are invalid (TypeError).
    """
    try:
        default_api_kwargs: dict[str, int | float | list[int] | str | bool] = {
            ROLLOUT_HORIZON_API_KWARG_NAME: api_config.default_api_kwargs.rollout_horizon,
            MC_DROPOUT_SAMPLES_API_KWARG_NAME: api_config.default_api_kwargs.mc_dropout_samples,
            CONFIDENCE_LEVEL_API_KWARG_NAME: api_config.default_api_kwargs.confidence_level,
            TIME_STEP_INCREMENT_API_KWARG_NAME: api_config.default_api_kwargs.time_step_increment,
            NUM_EVICTIONS_API_KWARG_NAME: api_config.default_api_kwargs.num_evictions,
            EXCLUDED_KEYS_API_KWARG_NAME: api_config.default_api_kwargs.excluded_keys,
            PROB_WEIGHT_API_KWARG_NAME: api_config.default_api_kwargs.prob_weight,
            CONF_WEIGHT_API_KWARG_NAME: api_config.default_api_kwargs.conf_weight,
            RETURN_ALL_SCORES_API_KWARG_NAME: api_config.default_api_kwargs.return_all_scores,
            RETURN_PROB_CONF_API_KWARG_NAME: api_config.default_api_kwargs.return_prob_conf,
        }

        debug(
            "Default API kwargs retrieval executed",
            extra={
                "api_kwargs_default": list(default_api_kwargs.keys()),
                "context": "Default API kwargs retrieval",
            },
        )

        return default_api_kwargs
    except (AttributeError, KeyError, TypeError) as e:
        error(
            "Default API kwargs retrieval failed",
            extra={
                "exception": str(e),
                "config_keys": list(api_config.keys())
                if hasattr(api_config, "keys")
                else None,
                "context": "Default API kwargs retrieval",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
