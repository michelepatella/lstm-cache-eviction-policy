import logging
from pathlib import Path
from typing import Literal

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------
# Configuration
# ----------------------------
CONFIG_DIRECTORY_PATH = PROJECT_ROOT / "src" / "pipeline" / "config"
CONFIG_FILE_NAME = "config.yaml"

CONFIG_SECTIONS_WITH_PARAMS = {
    "model": "params",
    "training.optimizer": "params",
}


# ----------------------------
# JSON
# ----------------------------
JSON_WRAP_BOX_DEFAULT = True
JSON_INDENT = 4


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_NAME = "phase"
LOGS_PHASE_DEFAULT = "unknown"

LOGS_CONFIGURATION_PHASE = "configuration"
LOGS_DATA_GENERATION_PHASE = "data generation"
LOGS_DATA_PREPROCESSING_PHASE = "data preprocessing"
LOGS_VALIDATION_PHASE = "validation"
LOGS_TRAINING_PHASE = "training"
LOGS_TESTING_PHASE = "testing"
LOGS_SIMULATIONS_PHASE = "simulations"

LOGS_FORMAT = "[%(phase)s] %(levelname)s: %(message)s"

LOGS_FILE_BASE_LEVEL = logging.DEBUG

LOGS_DIRECTORY_PATH = PROJECT_ROOT / "logs"

LOGS_INFO_DIRECTORY_PATH = LOGS_DIRECTORY_PATH / "info"
LOGS_DEBUG_DIRECTORY_PATH = LOGS_DIRECTORY_PATH / "debug"
LOGS_ERROR_DIRECTORY_PATH = LOGS_DIRECTORY_PATH / "error"

LOGS_INFO_FILE_PATH = LOGS_INFO_DIRECTORY_PATH / "info.log"
LOGS_DEBUG_FILE_PATH = LOGS_DEBUG_DIRECTORY_PATH / "debug.log"
LOGS_ERROR_FILE_PATH = LOGS_ERROR_DIRECTORY_PATH / "error.log"

LOGS_FILE_MAX_BYTES = 10_000_000
LOGS_FILE_BACKUP_COUNT = 1000


# ----------------------------
# Time
# ----------------------------
HOURS_IN_DAY = 24
SECONDS_IN_DAY = 86_400
SECONDS_IN_HOUR = 3_600
MICROSECONDS_IN_SECOND = 1_000_000


# ----------------------------
# Data Distributions
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
TIMESTAMP_COLUMN_NAME = "timestamp"
REQUEST_COLUMN_NAME = "request"
SIN_TIME_COLUMN_NAME = "sin_time"
COS_TIME_COLUMN_NAME = "cos_time"

DATASET_RAW_TYPE = "raw"
DATASET_PROCESSED_TYPE = "processed"

TRAINING_SPLIT_TYPE = "training"
TESTING_SPLIT_TYPE = "testing"

STATIC_RAW_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "static"
DYNAMIC_RAW_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "dynamic"
STATIC_PROCESSED_DATASET_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "static"
)
DYNAMIC_PROCESSED_DATASET_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "dynamic"
)

STATIC_RAW_DATASET_FILE_NAME = "static_raw_dataset.csv"
DYNAMIC_RAW_DATASET_FILE_NAME = "dynamic_raw_dataset.csv"
STATIC_PROCESSED_DATASET_FILE_NAME = "static_processed_dataset.csv"
DYNAMIC_PROCESSED_DATASET_FILE_NAME = "dynamic_processed_dataset.csv"


# ----------------------------
# Data Generation
# ----------------------------
DATA_GENERATION_INITIAL_HOUR = 0
DATA_GENERATION_FINAL_HOUR = 23

DATA_GENERATION_INITIAL_TIMESTAMP = 0.0
DATA_GENERATION_INITIAL_CURRENT_DAY = 0
DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY = 0.0


# ----------------------------
# Data Preprocessing
# ----------------------------
MISSING_VALUES_REMOVAL_DROPNA_AXIS = 0
MISSING_VALUES_REMOVAL_DROPNA_HOW: Literal["any", "all"] = "any"


# ----------------------------
# Hardware Devices
# ----------------------------
HW_DEVICES = ["cpu", "cuda", "mps"]


# ----------------------------
# Optimizers
# ----------------------------
ADAM_OPTIMIZER = "adam"
ADAMW_OPTIMIZER = "adamw"
SGD_OPTIMIZER = "sgd"

TRAINING_OPTIMIZERS = [
    ADAM_OPTIMIZER,
    ADAMW_OPTIMIZER,
    SGD_OPTIMIZER,
]


# ----------------------------
# Criterion
# ----------------------------
CLASS_WEIGHT_TYPE = "balanced"


# ----------------------------
# Model (LSTM)
# ----------------------------
LSTM_PARAM_NAMES = [
    "hidden_size",
    "num_layers",
    "bias",
    "batch_first",
    "dropout",
    "bidirectional",
    "proj_size",
]

TRAINED_MODEL_STATIC_DIRECTORY = PROJECT_ROOT / "models" / "static"
TRAINED_MODEL_DYNAMIC_DIRECTORY = PROJECT_ROOT / "models" / "dynamic"

TRAINED_MODEL_STATIC_FILE_NAME = "trained_static_model.pt"
TRAINED_MODEL_DYNAMIC_FILE_NAME = "trained_dynamic_model.pt"

TRAINING_MODEL_MODE = "train"
EVALUATION_MODEL_MODE = "eval"
MC_DROPOUT_MODEL_MODE = "mc_dropout"


# ----------------------------
# Training
# ----------------------------
GRID_SEARCH_DESC = "Grid Search"


# ----------------------------
# Training
# ----------------------------
TRAINING_EPOCHS_DESC = "Training"
TRAINING_SINGLE_EPOCH_DESC = "Epoch"


# ----------------------------
# Early Stopping
# ----------------------------
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_DISABLED = False


# ----------------------------
# Monte Carlo (MC) Dropout
# ----------------------------
MC_DROPOUT_DISABLED = False
MC_DROPOUT_ENABLED = True

MC_DROPOUT_NUM_SAMPLES_DEFAULT = 1

MC_DROPOUT_FLAG = "mc_dropout"

MC_DROPOUT_UNBIASED_VARIANCE = False


# ----------------------------
# Cache simulations
# ----------------------------
EVICTION_POLICY_API_ENDPOINT = "http://localhost:8000/evict"
EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME = "keys_in_cache"
EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME = "last_accesses"
EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME = "user_kwargs"


# ----------------------------
# Cache eviction policies
# ----------------------------
LSTM_CACHE_NAME = "LSTM"
FIFO_CACHE_NAME = "FIFO"
LFU_CACHE_NAME = "LFU"
LRU_CACHE_NAME = "LRU"
RANDOM_CACHE_NAME = "RANDOM"


# ----------------------------
# Model metrics
# ----------------------------
MODEL_COMPUTE_METRICS_DEFAULT = False
MODEL_METRICS_AVG_LOSS = "avg_loss"
MODEL_METRICS_CLASS_REPORT_NAME = "class_report"
MODEL_METRICS_TOP_K_ACCURACY_NAME = "top_k_accuracy"
MODEL_METRICS_COHEN_KAPPA_SCORE_NAME = "cohen_kappa_score"
MODEL_METRICS_ACCURACY_NAME = "accuracy"
MODEL_METRICS_MACRO_AVG_NAME = "macro avg"
MODEL_METRICS_WEIGHTED_AVG_NAME = "weighted avg"
MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT = True
MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION = 0

# ----------------------------
# Cache simulations metrics
# ----------------------------
HIT_COUNTER_NAME = "hit_count"
MISS_COUNTER_NAME = "miss_count"

POLICY_NAME = "policy"
HIT_RATE_NAME = "hit_rate"
MISS_RATE_NAME = "miss_rate"
EVICTION_MISTAKE_RATE_NAME = "eviction_mistake_rate"
AVG_CACHE_LATENCY_NAME = "avg_cache_latency"

TIMELINE_NAME = "timeline"
TIMELINE_INDEX_NAME = "index"
TIMELINE_INSTANT_HIT_RATE_NAME = "instant_hit_rate"


# ----------------------------
# Results
# ----------------------------
RESULTS_DIRECTORY_PATH = PROJECT_ROOT / "reports" / "results"

STATIC_MODEL_RESULTS_FILE_NAME = "static_model_results.json"
DYNAMIC_MODEL_RESULTS_FILE_NAME = "dynamic_model_results.json"

STATIC_SIMULATIONS_RESULTS_FILE_NAME = "static_simulations_results.json"
DYNAMIC_SIMULATIONS_RESULTS_FILE_NAME = "dynamic_simulations_results.json"


# ----------------------------
# Plots
# ----------------------------
PLOTS_DIRECTORY_PATH = PROJECT_ROOT / "reports" / "plots"
PLOT_SIZE = 30
PLOT_TITLE_FONT_SIZE = 18
PLOT_LABEL_FONT_SIZE = 16

ZIPF_LOG_LOG_PLOT_FILE_NAME = "zipf_log_log.png"
ZIPF_LOG_LOG_PLOT_MARKER = "o"
ZIPF_LOG_LOG_PLOT_TITLE = "Zipf Distribution (Log-Log)"
ZIPF_LOG_LOG_PLOT_X_LABEL = "Key"
ZIPF_LOG_LOG_PLOT_Y_LABEL = "Frequency"

DAILY_PROFILE_PLOT_FILE_NAME = "daily_profile.png"
DAILY_PROFILE_PLOT_STEP = 1
DAILY_PROFILE_PLOT_BIN_SIZE = 0.5
DAILY_PROFILE_PLOT_TITLE = "Distribution of Requests Over Day"
DAILY_PROFILE_PLOT_X_LABEL = "Hour of Day"
DAILY_PROFILE_PLOT_Y_LABEL = "Number of Requests"

KEY_USAGE_HEATMAP_FILE_NAME = "key_usage.png"
KEY_USAGE_HEATMAP_ROTATION = 45
KEY_USAGE_HEATMAP_STEP = 25
KEY_USAGE_HEATMAP_ASPECT: Literal["equal", "auto"] = "auto"
KEY_USAGE_HEATMAP_CMAP = "turbo"
KEY_USAGE_HEATMAP_TITLE = "Heatmap of Key Access Frequency by Hour of Day"
KEY_USAGE_HEATMAP_X_LABEL = "Key"
KEY_USAGE_HEATMAP_Y_LABEL = "Hour of Day"
KEY_USAGE_HEATMAP_COLORBAR_LABEL = "Access Count"

HIT_MISS_RATES_PLOT_FILE_NAME = "hit_miss_rates.png"
HIT_MISS_RATES_PLOT_PAD = 3.0
HIT_MISS_RATES_PLOT_NUM_ROWS = 2
HIT_MISS_RATES_PLOT_NUM_COLS = 1
HIT_RATE_SUBPLOT_LINE_STYLE = "--"
MISS_RATE_SUBPLOT_LINE_STYLE = "-"
HIT_RATE_SUBPLOT_TITLE = "Hit Rate Over Time"
MISS_RATE_SUBPLOT_TITLE = "Miss Rate Over Time"
HIT_MISS_RATE_SUBPLOT_X_LABEL = "Request"
HIT_RATE_SUBPLOT_Y_LABEL = "Hit Rate (%)"
MISS_RATE_SUBPLOT_Y_LABEL = "Miss Rate (%)"
HIT_MISS_RATES_SUBPLOTS_TITLE_NAME = "title"
HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME = "ylabel"
HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME = "line_style"
HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME = "transform"
HIT_MISS_RATES_SUBPLOTS = [
    {
        HIT_MISS_RATES_SUBPLOTS_TITLE_NAME: HIT_RATE_SUBPLOT_TITLE,
        HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME: HIT_RATE_SUBPLOT_Y_LABEL,
        HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME: HIT_RATE_SUBPLOT_LINE_STYLE,
        HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME: lambda hit: hit,
    },
    {
        HIT_MISS_RATES_SUBPLOTS_TITLE_NAME: MISS_RATE_SUBPLOT_TITLE,
        HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME: MISS_RATE_SUBPLOT_Y_LABEL,
        HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME: MISS_RATE_SUBPLOT_LINE_STYLE,
        HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME: lambda hit: 100 - hit,
    },
]
