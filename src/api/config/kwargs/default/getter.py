from box import Box
from fastapi import HTTPException, status

from api.const import (
    API_KWARGS_CONF_WEIGHT_NAME,
    API_KWARGS_CONFIDENCE_LEVEL_NAME,
    API_KWARGS_EXCLUDED_KEYS_NAME,
    API_KWARGS_MC_DROPOUT_SAMPLES_NAME,
    API_KWARGS_NUM_EVICTIONS_NAME,
    API_KWARGS_PROB_WEIGHT_NAME,
    API_KWARGS_RETURN_ALL_SCORES_NAME,
    API_KWARGS_RETURN_API_KWARGS_NAME,
    API_KWARGS_RETURN_PROB_CONF_NAME,
    API_KWARGS_ROLLOUT_HORIZON_NAME,
    API_KWARGS_TIME_STEP_INCREMENT_NAME,
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
            API_KWARGS_ROLLOUT_HORIZON_NAME: api_config.default_api_kwargs.rollout_horizon,
            API_KWARGS_MC_DROPOUT_SAMPLES_NAME: api_config.default_api_kwargs.mc_dropout_samples,
            API_KWARGS_CONFIDENCE_LEVEL_NAME: api_config.default_api_kwargs.confidence_level,
            API_KWARGS_TIME_STEP_INCREMENT_NAME: api_config.default_api_kwargs.time_step_increment,
            API_KWARGS_NUM_EVICTIONS_NAME: api_config.default_api_kwargs.num_evictions,
            API_KWARGS_EXCLUDED_KEYS_NAME: api_config.default_api_kwargs.excluded_keys,
            API_KWARGS_PROB_WEIGHT_NAME: api_config.default_api_kwargs.prob_weight,
            API_KWARGS_CONF_WEIGHT_NAME: api_config.default_api_kwargs.conf_weight,
            API_KWARGS_RETURN_ALL_SCORES_NAME: api_config.default_api_kwargs.return_all_scores,
            API_KWARGS_RETURN_PROB_CONF_NAME: api_config.default_api_kwargs.return_prob_conf,
            API_KWARGS_RETURN_API_KWARGS_NAME: api_config.default_api_kwargs.return_api_kwargs,
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
