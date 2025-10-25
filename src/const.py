from pathlib import Path

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_NESTED_ENABLED = True
MLFLOW_PYTORCH_SAVE_MODEL_PATH = "model"


# ----------------------------
# Dagshub
# ----------------------------
DAGS_HUB_MLFLOW_ENABLED = True
DAGS_HUB_REPO_NAME_ENV_VAR_NAME = "DAGS_HUB_REPO_NAME"
DAGS_HUB_REPO_OWNER_ENV_VAR_NAME = "DAGS_HUB_REPO_OWNER"


# ----------------------------
# Logs
# ----------------------------
LOGS_VALIDATION_PHASE = "validation"
LOGS_TRAINING_PHASE = "training"


# ----------------------------
# Data
# ----------------------------
DATA_DISTRIBUTION_STATIC_MODE = "static"
DATA_DISTRIBUTION_DYNAMIC_MODE = "dynamic"
DATA_DISTRIBUTION_MODES = [
    DATA_DISTRIBUTION_STATIC_MODE,
    DATA_DISTRIBUTION_DYNAMIC_MODE,
]

DATA_GENERATION_INITIAL_HOUR = 0
DATA_GENERATION_FINAL_HOUR = 23


# ----------------------------
# Dataset
# ----------------------------
DATASET_TIMESTAMP_COLUMN_NAME = "timestamp"
DATASET_REQUEST_COLUMN_NAME = "request"

DATASET_RAW_TYPE = "raw"

DATASET_TRAINING_SPLIT_TYPE = "training"
DATASET_TESTING_SPLIT_TYPE = "testing"


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
# Cache
# ----------------------------
CACHE_LSTM_NAME = "LSTM"


# ----------------------------
# Simulations Metrics
# ----------------------------
SIMULATIONS_METRICS_POLICY_NAME = "policy"
SIMULATIONS_METRICS_HIT_COUNTER_NAME = "hit_count"
SIMULATIONS_METRICS_MISS_COUNTER_NAME = "miss_count"
SIMULATIONS_METRICS_TIMELINE_NAME = "timeline"
