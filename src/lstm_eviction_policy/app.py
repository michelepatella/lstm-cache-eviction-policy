from typing import List, Any

from fastapi import FastAPI

from lstm_eviction_policy.initialization.initializer import initialize_api

app = FastAPI()

# Setup for API: Load model and device, using
# configuration settings
model, device = initialize_api()


@app.post("/evict")
def evict_key(keys_in_cache: List[Any], last_accesses: List):
    pass
