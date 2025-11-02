# ----------------------------
# Cache
# ----------------------------
CACHE_LSTM_NAME = "LSTM"


# ----------------------------
# Data
# ----------------------------
DATA_DISTRIBUTION_STATIC_MODE = "static"
DATA_DISTRIBUTION_DYNAMIC_MODE = "dynamic"
DATA_DISTRIBUTION_MODES = [
    DATA_DISTRIBUTION_STATIC_MODE,
    DATA_DISTRIBUTION_DYNAMIC_MODE,
]


# ----------------------------
# Dataset
# ----------------------------
DATASET_COLUMN_TIMESTAMP_NAME = "timestamp"
DATASET_COLUMN_REQUEST_NAME = "request"

DATASET_RAW_TYPE = "raw"

DATASET_TRAINING_SPLIT_TYPE = "training"
DATASET_TESTING_SPLIT_TYPE = "testing"


# ----------------------------
# HW Device
# ----------------------------
HW_DEVICE_NAMES = ["cpu", "cuda", "mps"]


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_VALIDATION = "validation"


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_NESTED = True


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
# Simulations Metrics
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
