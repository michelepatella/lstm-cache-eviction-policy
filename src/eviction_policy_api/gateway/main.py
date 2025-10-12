from typing import Any, Dict, List, Tuple

from fastapi import FastAPI

from eviction_policy_api.gateway.initialization.initializer import initialize_api
from eviction_policy_api.gateway.kwargs.builder import (
    build_api_kwargs,
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
    # Build API kwargs
    api_kwargs = build_api_kwargs(api_config, user_kwargs)
