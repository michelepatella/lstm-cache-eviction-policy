import requests
from box import Box
from fastapi import FastAPI

from components.json.io.loader import load_json
from eviction_policy_api.const import (
    API_CONFIG_FILE_PATH,
    API_GATEWAY_ENDPOINT,
    PREDICTOR_SERVICE_ENDPOINT,
    PREDICTOR_SERVICE_PARAMS,
)
from eviction_policy_api.gateway.kwargs.builder import (
    build_api_kwargs,
)
from eviction_policy_api.gateway.kwargs.default_getter import (
    get_default_kwargs,
)

app = FastAPI()

# Setup for API Gateway: Load API configuration
# meta data from JSON file
api_config = load_json(API_CONFIG_FILE_PATH)


@app.post(API_GATEWAY_ENDPOINT)
def evict_key(
    keys_in_cache: list[int],
    last_accesses: list[tuple[float, int]],
    user_kwargs: dict[str, int | float | list[int] | str | bool],
):
    # Retrieve default kwargs from
    # API configuration
    default_kwargs = get_default_kwargs(api_config)

    # Build API kwargs
    api_kwargs = build_api_kwargs(default_kwargs, user_kwargs)

    # Prepare params for autoregressive rollout service
    autoregressive_rollout_params = Box(PREDICTOR_SERVICE_PARAMS)
    autoregressive_rollout_params.last_accesses = last_accesses
    autoregressive_rollout_params.model_path = api_config.model.path
    autoregressive_rollout_params.model_params = api_config.model.params
    autoregressive_rollout_params.device_type = api_config.hardware.device_type
    autoregressive_rollout_params.min_key = api_config.model.keys.min
    autoregressive_rollout_params.max_key = api_config.model.keys.max
    autoregressive_rollout_params.num_features = api_config.model.num_features
    autoregressive_rollout_params.embedding_dim = (
        api_config.model.embedding_dim
    )
    autoregressive_rollout_params.rollout_horizon = api_kwargs.rollout_horizon
    autoregressive_rollout_params.mc_dropout_samples = (
        api_kwargs.mc_dropout_samples
    )
    autoregressive_rollout_params.time_step_increment = (
        api_kwargs.time_step_increment
    )

    # Call autoregressive rollout service
    autoregressive_rollout_response = requests.post(
        PREDICTOR_SERVICE_ENDPOINT,
        json=autoregressive_rollout_params.to_dict(),
    )
    autoregressive_rollout_response.raise_for_status()

    # Extract response of autoregressive rollout service
    autoregressive_rollout_result = Box(autoregressive_rollout_response.json())
    outputs = autoregressive_rollout_result.outputs
    variances = autoregressive_rollout_result.variances
