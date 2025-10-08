from typing import List, Any

from fastapi import FastAPI

from lstm_eviction_policy.initializer import initialize_api
from utils.model.initialization.trained_model_initializer import initialize_trained_model

app = FastAPI()

# Setup for API: Load model and device, using
# configuration settings
model, device = initialize_api()

@app.post("/evict")
def evict_key(keys_in_cache: List[Any], last_accesses: List[]):
