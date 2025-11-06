"""const.py

Module dedicated to defining fundamental constants and enumerations
specifically related to data characteristics, dataset column names,
hardware devices, and simulation metrics.

This module provides core definitions used across the entire project,
including data distribution modes, key dataset identifiers, logger phase
names, and the set of available optimizers and devices.
"""

# ----------------------------
# Cache
# ----------------------------
CACHE_LSTM_NAME = "LSTM"


# ----------------------------
# Data
# ----------------------------
DATA_STATIC_MODE = "static"
DATA_DYNAMIC_MODE = "dynamic"
DATA_REAL_MODE = "real"
DATA_MODES = [
    DATA_STATIC_MODE,
    DATA_DYNAMIC_MODE,
    DATA_REAL_MODE
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
HW_DEVICE_CPU_NAME = "cpu"
HW_DEVICE_CUDA_NAME = "cuda"
HW_DEVICE_MPS_NAME = "mps"
HW_DEVICE_NAMES = [HW_DEVICE_CPU_NAME, HW_DEVICE_CUDA_NAME, HW_DEVICE_MPS_NAME]


# ----------------------------
# Logs
# ----------------------------
LOGS_LOGGER_NAME = "logger"

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
