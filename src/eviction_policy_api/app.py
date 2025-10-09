from typing import List, Any

from fastapi import FastAPI

from eviction_policy_api.initialization.initializer import initialize_api

app = FastAPI()

# Setup for API: Load model, device and
# API configuration
model, device, api_config = initialize_api()


@app.post("/evict")
def evict_key(keys_in_cache: List[Any], last_accesses: List):
    pass
