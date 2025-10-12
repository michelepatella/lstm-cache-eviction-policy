from typing import Any, Dict, List, Tuple

from fastapi import FastAPI

from eviction_policy_api.initialization.initializer import initialize_api
from eviction_policy_api.utils.kwargs.api_kwargs_builder import (
    build_api_kwargs,
)
from eviction_policy_api.utils.kwargs.default_kwargs_getter import (
    get_default_kwargs,
)

app = FastAPI()

# Setup for API: Load model, device and
# API configuration
model, device, api_config = initialize_api()


@app.post("/evict")
def evict_key(
    keys_in_cache: List[Any],
    last_accesses: List[Tuple[float, Any]],
    user_kwargs: Dict[str, int | float | List[Any] | str | bool],
):
    # Retrieve default kwargs from
    # API configuration
    default_kwargs = get_default_kwargs(api_config)

    # Build API kwargs by merging default
    # and user-provided values
    api_kwargs = build_api_kwargs(default_kwargs, user_kwargs)
