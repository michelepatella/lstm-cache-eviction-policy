"""const.py

Module dedicated to defining constants and magic strings specifically
related to project structure, file paths, and high-level configuration.

This module includes paths for configuration files, results, and plots,
names of cache policies, logging phases, and specific simulation
metrics used throughout the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Cache
# ----------------------------
CACHE_LFU_NAME = "LFU"
CACHE_LRU_NAME = "LRU"
CACHE_RANDOM_NAME = "RANDOM"


# ----------------------------
# Config
# ----------------------------
CONFIG_FILE_PATH = PROJECT_ROOT / "src" / "pipeline" / "config" / "config.yaml"


# ----------------------------
# Dagshub
# ----------------------------
DAGS_HUB_DVC = True

DAGS_HUB_ENV_VAR_REPO_NAME = "DAGS_HUB_REPO"
DAGS_HUB_ENV_VAR_REPO_OWNER_NAME = "DAGS_HUB_REPO_OWNER"
DAGS_HUB_REPO_OWNER = os.getenv(DAGS_HUB_ENV_VAR_REPO_OWNER_NAME, "")
DAGS_HUB_REPO_NAME = os.getenv(DAGS_HUB_ENV_VAR_REPO_NAME, "")


# ----------------------------
# Dataset
# ----------------------------
DATASET_PROCESSED_TYPE = "processed"

DATASET_CHUNK_RESET_INDEX_DROP = True

DATASET_MISSING_VALUES_REMOVAL_DROPNA_HOWS = ["any", "all"]


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_DATA_PREPARATION = "data preparation"
LOGS_PHASE_DATA_PREPROCESSING = "data preprocessing"
LOGS_PHASE_TRAINING = "training"
LOGS_PHASE_TESTING = "testing"
LOGS_PHASE_SIMULATIONS = "simulations"

LOGS_LEVEL_NAMES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# ----------------------------
# Loss
# ----------------------------
LOSS_CLASS_WEIGHT_TYPES = [None, "balanced"]


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_ARTIFACT_PATH = "model"


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
MODEL_OPTIMIZATIONS_QUANTIZATION_ENGINE_NAMES = [
    "fbgemm",
    "qnnpack",
    "onednn",
    "xnnpack",
    "nnpi",
    "acl",
]


# ----------------------------
# Plot
# ----------------------------
PLOT_STATIC_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "static" / "static_zipf_log_log.png"
)
PLOT_STATIC_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "static" / "static_daily_profile.png"
)
PLOT_STATIC_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "static" / "static_key_usage.png"
)
PLOT_STATIC_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "static" / "static_hit_miss_rates.png"
)

PLOT_DYNAMIC_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "dynamic" / "dynamic_zipf_log_log.png"
)
PLOT_DYNAMIC_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "plots"
    / "dynamic"
    / "dynamic_daily_profile.png"
)
PLOT_DYNAMIC_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "dynamic" / "dynamic_key_usage.png"
)
PLOT_DYNAMIC_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "plots"
    / "dynamic"
    / "dynamic_hit_miss_rates.png"
)

PLOT_REAL_ZIPF_LOG_LOG_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "real" / "real_zipf_log_log.png"
)
PLOT_REAL_DAILY_PROFILE_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "real" / "real_daily_profile.png"
)
PLOT_REAL_KEY_USAGE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "real" / "real_key_usage.png"
)
PLOT_REAL_HIT_MISS_RATES_FILE_PATH = (
    PROJECT_ROOT / "reports" / "plots" / "real" / "real_hit_miss_rates.png"
)


# ----------------------------
# Results
# ----------------------------
RESULTS_STATIC_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "static"
    / "static_model_results.json"
)
RESULTS_DYNAMIC_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "dynamic"
    / "dynamic_model_results.json"
)
RESULTS_REAL_MODEL_FILE_PATH = (
    PROJECT_ROOT / "reports" / "results" / "real" / "real_model_results.json"
)


RESULTS_STATIC_SIMULATIONS_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "static"
    / "static_simulations_results.json"
)
RESULTS_DYNAMIC_SIMULATIONS_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "dynamic"
    / "dynamic_simulations_results.json"
)
RESULTS_REAL_SIMULATIONS_FILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "results"
    / "real"
    / "real_simulations_results.json"
)


# ----------------------------
# Simulations
# ----------------------------
SIMULATIONS_METRICS_HIT_RATE_NAME = "hit_rate"
SIMULATIONS_METRICS_MISS_RATE_NAME = "miss_rate"
SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME = "eviction_mistake_rate"
SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME = "avg_cache_latency"
SIMULATIONS_METRICS_BELADY_MIN_HIT_RATE_NAME = "belady_min_hit_rate"
SIMULATIONS_METRICS_BELADY_MIN_MISS_RATE_NAME = "belady_min_miss_rate"
