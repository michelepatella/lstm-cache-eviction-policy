"""const.py

Module dedicated to defining fundamental constants and enumerations
specifically related to data characteristics, dataset column names,
hardware devices, and simulation metrics.

This module provides core definitions used across the entire project,
including data distribution modes, key dataset identifiers, logger phase
names, and the set of available optimizers and devices.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# API
# ----------------------------
API_RESPONSE_FIELD_DATA_NAME = "data"
API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME = "keys_to_evict"


# ----------------------------
# Cache
# ----------------------------
CACHE_LSTM_NAME = "LSTM"


# ----------------------------
# Dagshub
# ----------------------------
DAGS_HUB_ENV_VAR_REPO_NAME = "DAGS_HUB_REPO"
DAGS_HUB_ENV_VAR_REPO_OWNER_NAME = "DAGS_HUB_REPO_OWNER"
DAGS_HUB_REPO_OWNER = os.getenv(DAGS_HUB_ENV_VAR_REPO_OWNER_NAME, "")
DAGS_HUB_REPO_NAME = os.getenv(DAGS_HUB_ENV_VAR_REPO_NAME, "")


# ----------------------------
# Data
# ----------------------------
DATA_STATIC_MODE = "static"
DATA_DYNAMIC_MODE = "dynamic"
DATA_REAL_MODE = "real"
DATA_MODES = [
    DATA_STATIC_MODE,
    DATA_DYNAMIC_MODE,
    DATA_REAL_MODE,
]


# ----------------------------
# Data Loader
# ----------------------------
DATA_LOADER_SHUFFLE_DEFAULT = False


# ----------------------------
# Dataset
# ----------------------------
DATASET_COLUMN_TIMESTAMP_NAME = "timestamp"
DATASET_COLUMN_REQUEST_NAME = "request"

DATASET_RAW_TYPE = "raw"

DATASET_TRAINING_SPLIT_TYPE = "training"
DATASET_TESTING_SPLIT_TYPE = "testing"


# ----------------------------
# Gateway API
# ----------------------------
GATEWAY_API_ENV_VAR_FULL_BASE_NAME = "GATEWAY_API_ENDPOINT_BASE_URL"
GATEWAY_API_ENV_VAR_ENDPOINT_NAME = "GATEWAY_API_ENDPOINT"
GATEWAY_API_ENDPOINT = os.getenv(GATEWAY_API_ENV_VAR_ENDPOINT_NAME, "")
GATEWAY_API_FULL_URL = (
    os.getenv(GATEWAY_API_ENV_VAR_FULL_BASE_NAME, "") + GATEWAY_API_ENDPOINT
)


# ----------------------------
# Logs
# ----------------------------
LOGS_LOGGER_NAME = "logger"

LOGS_PHASE_VALIDATION = "validation"


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_NESTED = True

MLFLOW_ENV_VAR_MODEL_NAME = "MLFLOW_MODEL_NAME"
MLFLOW_MODEL_NAME = os.getenv(MLFLOW_ENV_VAR_MODEL_NAME, "")

MLFLOW_MODEL_TAG_ENVIRONMENT = "environment"
MLFLOW_MODEL_TAG_DATA_MODE = "data_mode"
MLFLOW_MODEL_TAG_ENVIRONMENT_PRODUCTION = "production"
MLFLOW_MODEL_TAG_ENVIRONMENT_STAGING = "staging"


# ----------------------------
# Optimizer
# ----------------------------
OPTIMIZER_ADAM_NAME = "adam"
OPTIMIZER_ADAMW_NAME = "adamw"
OPTIMIZER_SGD_NAME = "sgd"
OPTIMIZER_NAMES = [
    OPTIMIZER_ADAM_NAME,
    OPTIMIZER_ADAMW_NAME,
    OPTIMIZER_SGD_NAME,
]


# ----------------------------
# Resources
# ----------------------------
RESOURCES_DEVICE_CPU_NAME = "cpu"
RESOURCES_DEVICE_CUDA_NAME = "cuda"
RESOURCES_DEVICE_MPS_NAME = "mps"
RESOURCES_DEVICE_NAMES = [
    RESOURCES_DEVICE_CPU_NAME,
    RESOURCES_DEVICE_CUDA_NAME,
    RESOURCES_DEVICE_MPS_NAME,
]


# ----------------------------
# Simulations
# ----------------------------
SIMULATIONS_METRICS_POLICY_NAME = "policy"
SIMULATIONS_METRICS_HIT_COUNTER_NAME = "hit_count"
SIMULATIONS_METRICS_MISS_COUNTER_NAME = "miss_count"
SIMULATIONS_METRICS_TIMELINE_NAME = "timeline"


# ----------------------------
# Time
# ----------------------------
TIME_START_HOUR = 0
TIME_END_HOUR = 23
