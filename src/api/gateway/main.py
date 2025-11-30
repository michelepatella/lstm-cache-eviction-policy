"""main.py

This module defines the central API Gateway for the cache eviction policy.

It orchestrates the end-to-end inference pipeline by sequentially calling four
distinct gRPC microservices: Featurizer, Predictor, Scorer, and Decider.
It also includes a basic health check for the API and a decorator to standardize
the response format.

Functions:
    lifespan(app: FastAPI):
        Context manager for API startup/shutdown tasks.
    construct_api_response(
        f: Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        Decorator for standardizing API responses.
    gateway_api(
        keys_in_cache: list[int],
        last_accesses: list[tuple[float, int]],
        user_api_kwargs: dict[str, int | float | list[int] | str | bool] | None
    ) -> dict[str, Any]:
        The main endpoint that executes the cache eviction policy pipeline.
    _index() -> dict[str, Any]:
        Provides a basic health check for the API.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from api.config.pydantic.api_config import APIConfig
from api.const import (
    API_CONFIG_FILE_PATH,
    API_DESCRIPTION,
    API_RESPONSE_FIELD_MESSAGE_NAME,
    API_RESPONSE_FIELD_METHOD_NAME,
    API_RESPONSE_FIELD_STATUS_CODE_NAME,
    API_RESPONSE_FIELD_TIMESTAMP_NAME,
    API_RESPONSE_FIELD_URL_NAME,
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
from src.const import (
    API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME,
    API_RESPONSE_FIELD_DATA_NAME,
    GATEWAY_API_ENDPOINT,
    LOGS_LOGGER_NAME,
)


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


def construct_api_response(
    f: Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Decorator to standardize the API response format.

    This function wraps an asynchronous/synchronous API endpoint function
    (f) and ensures that the final output adheres to a consistent JSON
    structure, including metadata (method, timestamp, URL) and centralized
    handling for HTTPException and unexpected errors.

    Args:
        f (Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]):
            The original API endpoint function to wrap.

    Returns:
        Callable[..., Awaitable[dict[str, Any]]]: The wrapped function that
                                                  returns a standardized response
                                                  dictionary.
    """

    @wraps(f)
    async def wrap(
        *args: Any,
        request: Request,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """The wrapper function that executes the endpoint and formats
        the response.

        This function attempts to execute the decorated function. If successful,
        it constructs the standard response using the result. If an exception occurs
        (HTTPException or a general Exception), it catches it and formats the output
        as an error response with appropriate status codes.

        Args:
            *args (Any): Positional arguments passed to the original function.
            request (Request): The FastAPI Request object, implicitly passed by
                               FastAPI.
            **kwargs (Any): Keyword arguments passed to the original function.

        Returns:
            dict[str, Any]: The standardized API response dictionary,
                            containing message, status code, data, and metadata.
        """
        # Construct a standard API response, also
        # managing exceptions raising in the API
        try:
            result = (
                await f(*args, request=request, **kwargs)
                if callable(f)
                else f(*args, **kwargs)
            )
        except HTTPException as e:
            # Response for error-specific cases
            result = {
                API_RESPONSE_FIELD_MESSAGE_NAME: str(e.detail),
                API_RESPONSE_FIELD_STATUS_CODE_NAME: e.status_code,
                API_RESPONSE_FIELD_DATA_NAME: {},
            }
        except Exception as e:
            # Response for general, unexpected error cases
            result = {
                API_RESPONSE_FIELD_MESSAGE_NAME: str(e),
                API_RESPONSE_FIELD_STATUS_CODE_NAME: HTTPStatus.INTERNAL_SERVER_ERROR,
                API_RESPONSE_FIELD_DATA_NAME: {},
            }

        # Construct final API response
        response = {
            API_RESPONSE_FIELD_MESSAGE_NAME: result.get(
                API_RESPONSE_FIELD_MESSAGE_NAME,
            ),
            API_RESPONSE_FIELD_METHOD_NAME: request.method,
            API_RESPONSE_FIELD_STATUS_CODE_NAME: result.get(
                API_RESPONSE_FIELD_STATUS_CODE_NAME,
            ),
            API_RESPONSE_FIELD_TIMESTAMP_NAME: datetime.now().isoformat(),
            API_RESPONSE_FIELD_URL_NAME: str(request.url),
        }
        if API_RESPONSE_FIELD_DATA_NAME in result:
            response[API_RESPONSE_FIELD_DATA_NAME] = result[
                API_RESPONSE_FIELD_DATA_NAME
            ]
        return response

    return wrap


@app.post(GATEWAY_API_ENDPOINT)
@construct_api_response
def gateway_api(
    keys_in_cache: list[int],
    last_accesses: list[tuple[float, int]],
    user_api_kwargs: dict[str, int | float | list[int] | str | bool] | None,
) -> dict[str, Any]:
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
        dict[str, Any]: A dictionary containing the standard API response structure.

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
        return {
            API_RESPONSE_FIELD_MESSAGE_NAME: HTTPStatus.OK.phrase,
            API_RESPONSE_FIELD_STATUS_CODE_NAME: HTTPStatus.OK,
            API_RESPONSE_FIELD_DATA_NAME: {
                API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME: keys_to_evict,
            },
        }

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


@app.get("/")
@construct_api_response
def _index() -> dict[str, Any]:
    """Basic health check for the API.

    This function represents a basic health check for the API
    status verification.

    Returns:
        dict[str, Any]: A dictionary containing the HTTP status and a message
                        indicating the API is operational.
    """
    return {
        API_RESPONSE_FIELD_MESSAGE_NAME: HTTPStatus.OK.phrase,
        API_RESPONSE_FIELD_STATUS_CODE_NAME: HTTPStatus.OK,
        API_RESPONSE_FIELD_DATA_NAME: {},
    }
