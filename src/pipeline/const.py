"""const.py

Module dedicated to defining constants and magic strings specifically
related to project structure, file paths, and high-level configuration.

This module includes paths for configuration files, results, and plots,
names of cache policies, logging phases, and specific simulation
metrics used throughout the project.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Cache
# ----------------------------
CACHE_LFU_NAME = "LFU"
CACHE_LRU_NAME = "LRU"


# ----------------------------
# Dagshub
# ----------------------------
DAGS_HUB_DVC = True


# ----------------------------
# Dataset
# ----------------------------
DATASET_PROCESSED_TYPE = "processed"

DATASET_RESET_INDEX_DROP = True

DATASET_MISSING_VALUES_REMOVAL_DROPNA_HOWS = ["any", "all"]


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_DATA_PREPARATION = "data preparation"
LOGS_PHASE_DATA_PREPROCESSING = "data preprocessing"
LOGS_PHASE_TRAINING = "training"
LOGS_PHASE_TESTING = "testing"
LOGS_PHASE_SIMULATIONS = "simulations"

# ----------------------------
# Loss
# ----------------------------
LOSS_CLASS_WEIGHT_TYPES = [None, "balanced"]
LOSS_REDUCTIONS = [None, "mean", "sum"]


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_ARTIFACT_PATH = "model"

MLFLOW_MODEL_TAG_DATA_MODE = "data_mode"
MLFLOW_MODEL_TAG_STATE = "state"
MLFLOW_MODEL_TAG_STATE_STAGING = "staging"
MLFLOW_MODEL_TAG_STATE_PROD = "prod"


# ----------------------------
# Model
# ----------------------------
MODEL_COMPUTE_METRICS_TESTING = True

MODEL_OPTIMIZATIONS_QUANTIZATION_DTYPES = [
    "qint8",
    "quint8",
    "float16",
    "bfloat16",
]

# ----------------------------
# Pipeline Config
# ----------------------------
PIPELINE_CONFIG_FILE_PATH = (
    PROJECT_ROOT / "src" / "pipeline" / "config" / "pipeline_config.yaml"
)


# ----------------------------
# Plot
# ----------------------------
PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "static" / "static_zipf_log_log.png"
)
PLOT_STATIC_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "static"
    / "static_daily_profile.png"
)
PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "static" / "static_key_usage.png"
)
PLOT_STATIC_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "static"
    / "static_hit_miss_rates.png"
)

PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "dynamic"
    / "dynamic_zipf_log_log.png"
)
PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "dynamic"
    / "dynamic_daily_profile.png"
)
PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "dynamic" / "dynamic_key_usage.png"
)
PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "dynamic"
    / "dynamic_hit_miss_rates.png"
)

PLOT_REAL_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "real" / "real_zipf_log_log.png"
)
PLOT_REAL_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "real" / "real_daily_profile.png"
)
PLOT_REAL_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "real" / "real_key_usage.png"
)
PLOT_REAL_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "real" / "real_hit_miss_rates.png"
)


# ----------------------------
# Simulations
# ----------------------------
SIMULATIONS_METRICS_HIT_RATE_NAME = "hit_rate"
SIMULATIONS_METRICS_MISS_RATE_NAME = "miss_rate"
SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME = "eviction_mistake_rate"
SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME = "avg_cache_latency"
