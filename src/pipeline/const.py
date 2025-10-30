from pathlib import Path

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Config
# ----------------------------
CONFIG_FILE_PATH = PROJECT_ROOT / "src" / "pipeline" / "config" / "config.yaml"


# ----------------------------
# Logs
# ----------------------------
LOGS_DATA_GENERATION_PHASE = "data generation"
LOGS_DATA_PREPROCESSING_PHASE = "data preprocessing"
LOGS_TESTING_PHASE = "testing"
LOGS_SIMULATIONS_PHASE = "simulations"


# ----------------------------
# HW Device
# ----------------------------
HW_DEVICE_NAMES = ["cpu", "cuda", "mps"]


# ----------------------------
# Dataset
# ----------------------------
DATASET_PROCESSED_TYPE = "processed"


# ----------------------------
# Model
# ----------------------------
MODEL_COMPUTE_METRICS_ENABLED = True


# ----------------------------
# Cache
# ----------------------------
CACHE_LFU_NAME = "LFU"
CACHE_LRU_NAME = "LRU"
CACHE_RANDOM_NAME = "RANDOM"


# ----------------------------
# Simulations Metrics
# ----------------------------
SIMULATIONS_METRICS_HIT_RATE_NAME = "hit_rate"
SIMULATIONS_METRICS_MISS_RATE_NAME = "miss_rate"
SIMULATIONS_METRICS_EVICTION_MISTAKE_RATE_NAME = "eviction_mistake_rate"
SIMULATIONS_METRICS_AVG_CACHE_LATENCY_NAME = "avg_cache_latency"


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
