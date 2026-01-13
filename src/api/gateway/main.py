"""main.py

This module defines the central API Gateway for the cache eviction policy.

It orchestrates the end-to-end inference pipeline by sequentially calling four
distinct gRPC microservices: Featurizer, Predictor, Scorer, and Decider.
The module also integrates Prometheus for real-time monitoring, and standardizes
all responses through a dedicated decorator.

Functions:
    lifespan(app: FastAPI):
        Context manager for API startup/shutdown tasks.
    construct_api_response(
        f: Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        Decorator for standardizing API responses.
    gateway_api(payload: GatewayAPIInput, request: Request) -> dict[str, Any]:
        The main endpoint that executes the cache eviction policy pipeline.
    _index(request: Request) -> dict[str, Any]:
        Provides a basic health check for the API.
    metrics() -> Response:
        Exposes Prometheus metrics for monitoring.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import (
    Instrumentator,
)
from prometheus_fastapi_instrumentator import (
    metrics as instr_metrics,
)

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
    PROMETHEUS_DECIDER_LATENCY_SECONDS_DOC,
    PROMETHEUS_DECIDER_LATENCY_SECONDS_NAME,
    PROMETHEUS_FEATURIZER_LATENCY_SECONDS_DOC,
    PROMETHEUS_FEATURIZER_LATENCY_SECONDS_NAME,
    PROMETHEUS_IN_PROGRESS_LABELS,
    PROMETHEUS_INCLUDE_IN_SCHEMA,
    PROMETHEUS_KEYS_EVICTED_RATIO_DOC,
    PROMETHEUS_KEYS_EVICTED_RATIO_NAME,
    PROMETHEUS_KEYS_EVICTED_TOT_DOC,
    PROMETHEUS_KEYS_EVICTED_TOT_NAME,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_RATIO_DOC,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_RATIO_NAME,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_TOT_DOC,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_TOT_NAME,
    PROMETHEUS_PREDICTOR_LATENCY_SECONDS_DOC,
    PROMETHEUS_PREDICTOR_LATENCY_SECONDS_NAME,
    PROMETHEUS_SCORER_LATENCY_SECONDS_DOC,
    PROMETHEUS_SCORER_LATENCY_SECONDS_NAME,
    PROMETHEUS_SHOULD_GZIP,
    PROMETHEUS_SHOULD_IGNORE_UNTEMPLATED,
    PROMETHEUS_SHOULD_INSTRUMENT_REQUESTS_IN_PROGRESS,
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
from api.gateway.schemas import GatewayAPIInput
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml
from const import (
    API_METRICS_ENDPOINT,
    API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME,
    API_RESPONSE_FIELD_DATA_NAME,
    GATEWAY_API_ENDPOINT,
    LOGS_LOGGER_NAME,
)

# Custom Prometheus metrics
KEYS_EVICTED_TOT = Counter(
    PROMETHEUS_KEYS_EVICTED_TOT_NAME,
    PROMETHEUS_KEYS_EVICTED_TOT_DOC,
)
KEYS_EVICTED_RATIO = Gauge(
    PROMETHEUS_KEYS_EVICTED_RATIO_NAME,
    PROMETHEUS_KEYS_EVICTED_RATIO_DOC,
)
KEYS_EXCLUDED_FROM_EVICTION_TOT = Counter(
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_TOT_NAME,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_TOT_DOC,
)
KEYS_EXCLUDED_FROM_EVICTION_RATIO = Gauge(
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_RATIO_NAME,
    PROMETHEUS_KEYS_EXCLUDED_FROM_EVICTION_RATIO_DOC,
)
FEATURIZER_LATENCY_SECONDS = Histogram(
    PROMETHEUS_FEATURIZER_LATENCY_SECONDS_NAME,
    PROMETHEUS_FEATURIZER_LATENCY_SECONDS_DOC,
)
PREDICTOR_LATENCY_SECONDS = Histogram(
    PROMETHEUS_PREDICTOR_LATENCY_SECONDS_NAME,
    PROMETHEUS_PREDICTOR_LATENCY_SECONDS_DOC,
)
SCORER_LATENCY_SECONDS = Histogram(
    PROMETHEUS_SCORER_LATENCY_SECONDS_NAME,
    PROMETHEUS_SCORER_LATENCY_SECONDS_DOC,
)
DECIDER_LATENCY_SECONDS = Histogram(
    PROMETHEUS_DECIDER_LATENCY_SECONDS_NAME,
    PROMETHEUS_DECIDER_LATENCY_SECONDS_DOC,
)

# Standard Prometheus metrics
instrumentator = Instrumentator(
    should_ignore_untemplated=PROMETHEUS_SHOULD_IGNORE_UNTEMPLATED,
    should_instrument_requests_inprogress=PROMETHEUS_SHOULD_INSTRUMENT_REQUESTS_IN_PROGRESS,
    excluded_handlers=[API_METRICS_ENDPOINT],
    inprogress_labels=PROMETHEUS_IN_PROGRESS_LABELS,
)
instrumentator.add(instr_metrics.request_size())
instrumentator.add(instr_metrics.response_size())
instrumentator.add(instr_metrics.latency())
instrumentator.add(instr_metrics.requests())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the application lifespan events (startup and shutdown).

    During startup:
        1. Loads the API configuration from the specified YAML file path,
           and validates and parses the configuration into the Pydantic model,
           storing it.
        2. Initializes the global logging system, setting the level defined
           in the configuration and configuring the handler.
        3. Instruments the FastAPI application for Prometheus metrics collection.

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

    # -----------------------------------------------
    # (Startup) 3. API instrumentation
    # -----------------------------------------------
    instrumentator.instrument(app).expose(
        app,
        include_in_schema=PROMETHEUS_INCLUDE_IN_SCHEMA,
        should_gzip=PROMETHEUS_SHOULD_GZIP,
    )

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
            **kwargs (Any): Keyword arguments passed to the original function.

        Returns:
            dict[str, Any]: The standardized API response dictionary,
                            containing message, status code, data, and metadata.
        """
        # Construct a standard API response, also
        # managing exceptions raising in the API
        request = next(
            (
                arg
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Request)
            ),
            None,
        )
        try:
            result = (
                await f(*args, **kwargs) if callable(f) else f(*args, **kwargs)
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
async def gateway_api(
    payload: GatewayAPIInput,
    request: Request,
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
        payload (GatewayAPIInput):
            The request body containing all input parameters, validated
            by the Pydantic model. This includes:
            - keys_in_cache (list[int]): Keys currently in the cache.
            - last_accesses (list[tuple[float, int]]): Recent access history.
            - user_api_kwargs (dict | None): Optional runtime configuration
                                             overrides.
       request (Request): FastAPI request object, required by the decorator.

    Returns:
        dict[str, Any]: A dictionary containing the standard API response
                        structure.

    Raises:
        HTTPException: If any API step fails.
    """
    try:
        # Extract payload
        keys_in_cache = payload.keys_in_cache
        last_accesses = payload.last_accesses
        user_api_kwargs = payload.user_api_kwargs

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
        with FEATURIZER_LATENCY_SECONDS.time():
            features, keys_seq, features_shape, keys_shape = (
                call_featurizer_service(last_accesses)
            )

        # ---------------------------------------------------------------
        # 2. Predictor Service: Make confidence-aware model predictions
        # ---------------------------------------------------------------
        with PREDICTOR_LATENCY_SECONDS.time():
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
        with SCORER_LATENCY_SECONDS.time():
            key_scores = call_scorer_service(outputs, variances, api_config)

        # ---------------------------------------------------------------
        # 4. Decider Service: Determine key(s) to evict
        # ---------------------------------------------------------------
        with DECIDER_LATENCY_SECONDS.time():
            keys_to_evict = call_decider_service(
                keys_in_cache,
                key_scores,
                api_config.kwargs.excluded_keys.value,
                api_config.kwargs.num_evictions.value,
            )

        # Update custom Prometheus metrics
        KEYS_EVICTED_TOT.inc(len(keys_to_evict))
        KEYS_EVICTED_RATIO.set(len(keys_to_evict) / len(keys_in_cache))
        KEYS_EXCLUDED_FROM_EVICTION_TOT.inc(
            len(api_config.kwargs.excluded_keys.value),
        )
        KEYS_EXCLUDED_FROM_EVICTION_RATIO.set(
            len(api_config.kwargs.excluded_keys.value) / len(keys_in_cache),
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
async def _index(request: Request) -> dict[str, Any]:
    """Basic health check for the API.

    This function represents a basic health check for the API
    status verification.

    Args:
        request (Request): FastAPI request object automatically
                           injected by the framework.

    Returns:
        dict[str, Any]: A dictionary containing the HTTP status and a message
                        indicating the API is operational.
    """
    return {
        API_RESPONSE_FIELD_MESSAGE_NAME: HTTPStatus.OK.phrase,
        API_RESPONSE_FIELD_STATUS_CODE_NAME: HTTPStatus.OK,
        API_RESPONSE_FIELD_DATA_NAME: {},
    }


@app.get(API_METRICS_ENDPOINT)
def metrics() -> Response:
    """Exposes global Prometheus metrics.

    This endpoint aggregates both custom application metrics and standard
    system metrics, providing them in the Prometheus text-based format.

    Returns:
        Response: A FastAPI Response object containing the plain-text metrics
                  buffer with the appropriate Content-Type.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
