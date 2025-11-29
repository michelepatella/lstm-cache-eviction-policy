"""main.py

This module defines the central API Gateway for the cache eviction policy.

It orchestrates the end-to-end inference pipeline by sequentially calling four
distinct gRPC microservices: Featurizer, Predictor, Scorer, and Decider.

Functions:
    lifespan(app: FastAPI):
        Context manager for API startup/shutdown tasks.
    gateway_api(
        keys_in_cache: list[int],
        last_accesses: list[tuple[float, int]],
        user_api_kwargs: dict[str, int | float | list[int] | str | bool] | None,
    ) -> list[int]:
        The main endpoint that executes the cache eviction policy.
"""

import logging
from contextlib import asynccontextmanager
from copy import deepcopy

from fastapi import FastAPI, HTTPException, status

from api.config.pydantic.api_config import APIConfig
from api.const import (
    API_CONFIG_FILE_PATH,
    API_DESCRIPTION,
    API_TITLE,
    LOGS_PHASE_API,
)
from api.gateway.callers.decider_service_caller import call_decider_service
from api.gateway.callers.featurizer_service_caller import (
    call_featurizer_service,
)
from api.gateway.callers.predictor_service_caller import (
    call_predictor_service,
)
from api.gateway.callers.scorer_service_caller import (
    call_scorer_service,
)
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml
from src.const import GATEWAY_API_ENDPOINT, LOGS_LOGGER_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the application lifespan events (startup and shutdown).

    During startup:
        1. Loads the API configuration from the specified YAML file path,
           and validates and parses the configuration into the Pydantic model,
           storing it.
        2. Initializes the global logging system, setting the level defined
           in the configuration and configuring the handler.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # -----------------------------------------------
    # (Startup) 1. API configuration preparation
    # -----------------------------------------------
    app.state.api_config_file = load_yaml(API_CONFIG_FILE_PATH)
    app.state.api_config = APIConfig(**app.state.api_config_file)

    # -----------------------------------------------
    # (Startup) 2. API logs initialization
    # -----------------------------------------------
    initialize_logs(
        logging.getLevelName(app.state.api_config.logs.level),
        GrafanaLokiHandler(),
    )
    logs_phase.set(LOGS_PHASE_API)

    yield


# Define the API
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.post(GATEWAY_API_ENDPOINT)
def gateway_api(
    keys_in_cache: list[int],
    last_accesses: list[tuple[float, int]],
    user_api_kwargs: dict[str, int | float | list[int] | str | bool] | None,
) -> list[int]:
    """The main endpoint executing the cache eviction policy.

    This function orchestrates a sequential process by calling the
    Featurizer, Predictor, Scorer, and Decider gRPC services.
    The pipeline steps are:
        1. Featurizer: Converts raw data into model-ready features.
        2. Predictor: Generates model outputs.
        3. Scorer: Calculates a score for each key based on model results.
        4. Decider: Selects the keys to evict.

    Args:
        keys_in_cache (list[int]): List of keys currently stored in the cache.
        last_accesses (list[tuple[float, int]]): A sequence of (hours in day, key)
                                                 tuples representing recent
                                                 cache accesses.
        user_api_kwargs (dict[str, int | float | list[int] | str | bool] | None):
            Optional dictionary of runtime arguments to override API
            default configurations.

    Returns:
        list[int]: A list of keys that should be evicted from the cache.

    Raises:
        HTTPException: If any API step fails.
    """
    try:
        info(
            "Gateway API started",
            extra={
                "keys_in_cache_num": len(keys_in_cache),
                "last_accesses_num": len(last_accesses),
                "api_kwargs_user": user_api_kwargs
                if user_api_kwargs
                else None,
                "context": "Gateway API",
            },
        )

        # ---------------------------------------------------------------
        # 0. Preparation: Merge user API kwargs with default ones
        # ---------------------------------------------------------------
        api_config = deepcopy(app.state.api_config)
        if user_api_kwargs is not None:
            api_config.kwargs = api_config.merge_api_kwargs(
                user_api_kwargs,
            )

        # ---------------------------------------------------------------
        # 1. Featurizer Service: Build features from raw data
        # ---------------------------------------------------------------
        features, keys_seq, features_shape, keys_shape = (
            call_featurizer_service(last_accesses)
        )

        # ---------------------------------------------------------------
        # 2. Predictor Service: Make confidence-aware model predictions
        # ---------------------------------------------------------------
        outputs, variances = call_predictor_service(
            features,
            keys_seq,
            features_shape,
            keys_shape,
            api_config,
        )

        # ---------------------------------------------------------------
        # 3. Scorer Service: Calculate key scores based on model results
        # ---------------------------------------------------------------
        key_scores = call_scorer_service(outputs, variances, api_config)

        # ---------------------------------------------------------------
        # 4. Decider Service: Determine key(s) to evict
        # ---------------------------------------------------------------
        keys_to_evict = call_decider_service(
            keys_in_cache,
            key_scores,
            api_config.kwargs.excluded_keys.value,
            api_config.kwargs.num_evictions.value,
        )

        info(
            "Gateway API completed",
            extra={
                "keys_to_evict_num": len(keys_to_evict),
                "context": "Gateway API",
            },
        )

        # Final response to the client
        return keys_to_evict

    except Exception as e:
        error(
            "Gateway API failed",
            extra={
                "exception": str(e),
                "keys_in_cache_num": len(keys_in_cache),
                "last_accesses_num": len(last_accesses),
                "api_kwargs_user": user_api_kwargs
                if user_api_kwargs
                else None,
                "context": "Gateway API",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    finally:
        # Async flush logs
        for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
            if isinstance(handler, GrafanaLokiHandler):
                handler.flush_buffer_async()
