from typing import Any

from box import Box
from fastapi import FastAPI, HTTPException, status

from api.config.kwargs.builder import (
    build_api_kwargs,
)
from api.config.kwargs.default.getter import (
    get_default_api_kwargs,
)
from api.const import (
    API_CONFIG_FILE_PATH,
    GATEWAY_API_ENDPOINT,
    GATEWAY_API_RETURN_API_KWARGS_NAME,
    GATEWAY_API_RETURN_CONF_MATRIX_NAME,
    GATEWAY_API_RETURN_KEY_SCORES_NAME,
    GATEWAY_API_RETURN_KEYS_TO_EVICT_NAME,
    GATEWAY_API_RETURN_PROB_MATRIX_NAME,
)
from api.gateway.callers.predictor_service_caller import (
    call_predictor_service,
)
from api.gateway.callers.scorer_service_caller import (
    call_scorer_service,
)
from components.caches.implementations.items.evictions.score_based_evictor import (
    evict_score_based_items,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml

app = FastAPI()

# Setup for Gateway API: Load API configuration
api_config = load_yaml(API_CONFIG_FILE_PATH)
api_config = Box(api_config)


@app.post(GATEWAY_API_ENDPOINT)
def gateway_api(
    keys_in_cache: list[int],
    last_accesses: list[tuple[float, int]],
    user_api_kwargs: dict[str, int | float | list[int] | str | bool],
) -> dict[str, Any]:
    """Gateway API endpoint for eviction decision.

    This endpoint receives the current cache state, last access
    information, and optional user-defined API kwargs. It orchestrates
    the predictor and scorer services and determines which keys
    should be evicted.

    Args:
        keys_in_cache (list[int]): List of keys currently in cache.
        last_accesses (list[tuple[float, int]]): List of (timestamp, key)
                                                 tuples.
        user_api_kwargs (dict[str, int|float|list[int]|str|bool]):
            Optional user-defined API kwargs overriding default API settings.

    Returns:
        dict[str, Any]: Dictionary containing:
            - keys_to_evict: Keys selected for eviction.
            - api_kwargs: API kwargs used to make predictions.
            - key_scores: Computed scores (Optional).
            - prob_matrix: Probability matrix (Optional).
            - conf_matrix: Confidence matrix (Optional).

    Raises:
        HTTPException: If any expected error occurs.
    """
    try:
        info(
            "Gateway API started",
            extra={
                "keys_in_cache_num": len(keys_in_cache),
                "last_accesses_num": len(last_accesses),
                "api_kwargs_user": list(user_api_kwargs.keys())
                if user_api_kwargs
                else None,
                "context": "Gateway API",
            },
        )

        # Retrieve default API kwargs from
        # API configuration
        default_api_kwargs = get_default_api_kwargs(api_config)

        # Build final API kwargs by merging
        # default and user-specified ones (if any)
        api_kwargs = build_api_kwargs(default_api_kwargs, user_api_kwargs)

        # Invoke predictor service to run autoregressive rollout
        # and get outputs and corresponding variances
        outputs, variances = call_predictor_service(
            last_accesses,
            api_kwargs,
            api_config,
        )

        # Invoke scorer service to calculate key scores
        # based on the probability of being used as well as
        # predictions confidence
        key_scores, prob_matrix, conf_matrix = call_scorer_service(
            outputs,
            variances,
            api_kwargs,
        )

        # Decide which keys to be evicted
        excluded_keys = api_kwargs.excluded_keys
        num_evictions = api_kwargs.num_evictions
        keys_to_evict = evict_score_based_items(
            keys_in_cache,
            key_scores,
            excluded_keys,
            num_evictions,
        )

        # Prepare response
        response: dict = {
            GATEWAY_API_RETURN_KEYS_TO_EVICT_NAME: keys_to_evict,
        }
        if api_kwargs.return_api_kwargs:
            response[GATEWAY_API_RETURN_API_KWARGS_NAME] = api_kwargs.__dict__
        if api_kwargs.return_all_scores:
            response[GATEWAY_API_RETURN_KEY_SCORES_NAME] = key_scores
        if api_kwargs.return_prob_conf:
            response[GATEWAY_API_RETURN_PROB_MATRIX_NAME] = prob_matrix
            response[GATEWAY_API_RETURN_CONF_MATRIX_NAME] = conf_matrix

        info(
            "Gateway API completed",
            extra={
                "keys_to_evict_num": len(keys_to_evict),
                "api_kwargs": api_kwargs.__dict__,
                "context": "Gateway API",
            },
        )

        return response
    except Exception as e:
        error(
            "Gateway API failed",
            extra={
                "exception": str(e),
                "keys_in_cache_num": len(keys_in_cache)
                if keys_in_cache
                else 0,
                "last_accesses_num": len(last_accesses)
                if last_accesses
                else 0,
                "api_kwargs_user": list(user_api_kwargs.keys())
                if user_api_kwargs
                else None,
                "context": "Gateway API",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
