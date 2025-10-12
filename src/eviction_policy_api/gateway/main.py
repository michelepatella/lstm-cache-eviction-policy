from typing import Any, Dict, List, Tuple

import requests
from box import Box
from fastapi import FastAPI

from eviction_policy_api.const import AUTOREGRESSIVE_ROLLOUT_SERVICE_ENDPOINT, AUTOREGRESSIVE_ROLLOUT_SERVICE_PARAMS, \
    META_DATA_CONFIG_FILE_PATH
from eviction_policy_api.gateway.kwargs.utils.default_kwargs_getter import get_default_kwargs
from utils.json.loader import load_json
from eviction_policy_api.gateway.kwargs.builder import (
    build_api_kwargs,
)

app = FastAPI()

# Setup for API Gateway: Load API configuration
# meta data from JSON file
api_config = load_json(META_DATA_CONFIG_FILE_PATH)


@app.post("/evict")
def evict_key(
    keys_in_cache: List[Any],
    last_accesses: List[Tuple[float, Any]],
    user_kwargs: Dict[str, int | float | List[Any] | str | bool],
):
    # Retrieve default kwargs from
    # API configuration
    default_kwargs = get_default_kwargs(api_config)

    # Build API kwargs
    api_kwargs = build_api_kwargs(default_kwargs, user_kwargs)

    # Prepare params for autoregressive rollout service
    autoregressive_rollout_params = Box(AUTOREGRESSIVE_ROLLOUT_SERVICE_PARAMS)
    autoregressive_rollout_params.model_path = api_config.model.path
    autoregressive_rollout_params.device_type = api_config.hardware.device_type
    autoregressive_rollout_params.last_accesses = last_accesses
    autoregressive_rollout_params.rollout_horizon = api_kwargs.rollout_horizon
    autoregressive_rollout_params.mc_dropout_samples = api_kwargs.mc_dropout_samples
    autoregressive_rollout_params.time_step_increment = api_kwargs.time_step_increment

    # Call autoregressive rollout service
    autoregressive_rollout_response = requests.post(
        AUTOREGRESSIVE_ROLLOUT_SERVICE_ENDPOINT,
        json=autoregressive_rollout_params.to_dict(),
    )
    autoregressive_rollout_response.raise_for_status()

    # Extract response of autoregressive rollout service
    autoregressive_rollout_result = Box(autoregressive_rollout_response.json())
    outputs = autoregressive_rollout_result.outputs
    variances = autoregressive_rollout_result.variances
