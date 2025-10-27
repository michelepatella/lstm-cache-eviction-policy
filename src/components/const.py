import logging
from pathlib import Path
from typing import Literal

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Logs
# ----------------------------
LOGS_LOGGER_NAME = "logger"

LOGS_PHASE_NAME = "phase"
LOGS_DEFAULT_PHASE = "unknown"

LOGS_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s %(LOGS_PHASE_NAME)s"
)

LOGS_STANDARD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}

LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME = "ELASTIC_ENDPOINT"
LOGS_ELASTIC_TOKEN_ENV_VAR_NAME = "ELASTIC_TOKEN"
LOGS_ELASTIC_INDEX_NAME_ENV_VAR_NAME = "ELASTIC_INDEX"

LOGS_ELASTIC_TIMESTAMP_FIELD_NAME = "timestamp"
LOGS_ELASTIC_LEVEL_FIELD_NAME = "level"
LOGS_ELASTIC_LOGGER_FIELD_NAME = "logger"
LOGS_ELASTIC_MESSAGE_FIELD_NAME = "message"

LOGS_INFO_FILE_PATH = PROJECT_ROOT / "logs" / "info" / "info.log"
LOGS_DEBUG_FILE_PATH = PROJECT_ROOT / "logs" / "debug" / "debug.log"
LOGS_ERROR_FILE_PATH = PROJECT_ROOT / "logs" / "error" / "error.log"

LOGS_FILE_BASE_LEVEL = logging.DEBUG
LOGS_FILE_MAX_BYTES = 10_000_000
LOGS_FILE_BACKUP_COUNT = 1000


# ----------------------------
# JSON
# ----------------------------
JSON_WRAP_BOX_ENABLED = True
JSON_INDENT = 4


# ----------------------------
# Time
# ----------------------------
TIME_HOURS_IN_DAY = 24
TIME_SECONDS_IN_DAY = 86_400
TIME_SECONDS_IN_HOUR = 3_600
TIME_MICROSECONDS_IN_SECOND = 1_000_000


# ----------------------------
# Data
# ----------------------------
DATA_GENERATION_INITIAL_TIMESTAMP = 0.0
DATA_GENERATION_INITIAL_CURRENT_DAY = 0
DATA_GENERATION_INITIAL_CURRENT_SECONDS_IN_DAY = 0.0


# ----------------------------
# Dataset
# ----------------------------
DATASET_SIN_TIME_COLUMN_NAME = "sin_time"
DATASET_COS_TIME_COLUMN_NAME = "cos_time"

DATASET_SIN_TIME_COLUMN_IDX = 0
DATASET_COS_TIME_COLUMN_IDX = 1
DATASET_TARGET_COLUMN_IDX = -1

DATASET_INDEX_DISABLED = False

DATASET_STATIC_RAW_FILE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "static" / "static_raw_dataset.csv"
)
DATASET_DYNAMIC_RAW_FILE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "dynamic" / "dynamic_raw_dataset.csv"
)
DATASET_STATIC_PROCESSED_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "static"
    / "static_processed_dataset.csv"
)
DATASET_DYNAMIC_PROCESSED_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic"
    / "dynamic_processed_dataset.csv"
)


# ----------------------------
# Missing Values Removal Dropna
# ----------------------------
MISSING_VALUES_REMOVAL_DROPNA_AXIS = 0
MISSING_VALUES_REMOVAL_DROPNA_HOW: Literal["any", "all"] = "any"


# ----------------------------
# Criterion
# ----------------------------
CRITERION_CLASS_WEIGHT_TYPE = "balanced"


# ----------------------------
# Model
# ----------------------------
MODEL_PARAM_NAMES = [
    "hidden_size",
    "num_layers",
    "bias",
    "batch_first",
    "dropout",
    "bidirectional",
    "proj_size",
]

MODEL_TRAINING_MODE = "train"
MODEL_EVALUATION_MODE = "eval"
MODEL_MC_DROPOUT_MODE = "mc_dropout"

MODEL_TRAINED_STATIC_FILE_PATH = (
    PROJECT_ROOT / "models" / "static" / "trained_static_model.pt"
)
MODEL_TRAINED_DYNAMIC_FILE_PATH = (
    PROJECT_ROOT / "models" / "dynamic" / "trained_dynamic_model.pt"
)

MODEL_COMPUTE_METRICS_DISABLED = False

MODEL_METRICS_AVG_LOSS_NAME = "avg_loss"
MODEL_METRICS_CLASS_REPORT_NAME = "class_report"
MODEL_METRICS_TOP_K_ACCURACY_NAME = "top_k_accuracy"
MODEL_METRICS_COHEN_KAPPA_SCORE_NAME = "cohen_kappa_score"
MODEL_METRICS_ACCURACY_NAME = "accuracy"
MODEL_METRICS_MACRO_AVG_NAME = "macro avg"
MODEL_METRICS_WEIGHTED_AVG_NAME = "weighted avg"
MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT_ENABLED = True
MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION = 0


# ----------------------------
# Early Stopping
# ----------------------------
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_DISABLED = False


# ----------------------------
# Grid Search
# ----------------------------
GRID_SEARCH_DESC = "Grid Search"


# ----------------------------
# Training
# ----------------------------
TRAINING_EPOCHS_DESC = "Training"
TRAINING_SINGLE_EPOCH_DESC = "Epoch"


# ----------------------------
# Monte Carlo (MC) Dropout
# ----------------------------
MC_DROPOUT_DISABLED = False
MC_DROPOUT_ENABLED = True
MC_DROPOUT_NUM_SAMPLES_DEFAULT = 1
MC_DROPOUT_FLAG_NAME = "mc_dropout"
MC_DROPOUT_UNBIASED_VARIANCE_DISABLED = False


# ----------------------------
# Autoregressive Rollout
# ----------------------------
AUTOREGRESSIVE_ROLLOUT_LAST_TIME_IDX = -1
AUTOREGRESSIVE_ROLLOUT_LAST_TIME_BATCH_IDX = 0


# ----------------------------
# Tensor
# ----------------------------
TENSOR_OUTPUTS_BATCH_DIM = 0
TENSOR_TEMPORAL_DIM = 1
TENSOR_CLASS_DIM = 1


# ----------------------------
# Eviction Policy API
# ----------------------------
EVICTION_POLICY_API_ENDPOINT = "http://localhost:8000/evict"
EVICTION_POLICY_API_KEYS_IN_CACHE_PARAM_NAME = "keys_in_cache"
EVICTION_POLICY_API_LAST_ACCESSES_PARAM_NAME = "last_accesses"
EVICTION_POLICY_API_USER_KWARGS_PARAM_NAME = "user_kwargs"


# ----------------------------
# Simulations Metrics
# ----------------------------
SIMULATIONS_METRICS_TIMELINE_INDEX_NAME = "index"
SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME = "instant_hit_rate"


# ----------------------------
# Plot
# ----------------------------
PLOT_SIZE = 30
PLOT_TITLE_FONT_SIZE = 18
PLOT_LABEL_FONT_SIZE = 16

PLOT_ZIPF_LOG_LOG_MARKER = "o"
PLOT_ZIPF_LOG_LOG_TITLE = "Zipf Distribution (Log-Log)"
PLOT_ZIPF_LOG_LOG_X_LABEL = "Key"
PLOT_ZIPF_LOG_LOG_Y_LABEL = "Frequency"

PLOT_DAILY_PROFILE_STEP = 1
PLOT_DAILY_PROFILE_BIN_SIZE = 0.5
PLOT_DAILY_PROFILE_TITLE = "Distribution of Requests Over Day"
PLOT_DAILY_PROFILE_X_LABEL = "Hour of Day"
PLOT_DAILY_PROFILE_Y_LABEL = "Number of Requests"

PLOT_KEY_USAGE_HEATMAP_ROTATION = 45
PLOT_KEY_USAGE_HEATMAP_STEP = 25
PLOT_KEY_USAGE_HEATMAP_ASPECT: Literal["equal", "auto"] = "auto"
PLOT_KEY_USAGE_HEATMAP_CMAP = "turbo"
PLOT_KEY_USAGE_HEATMAP_TITLE = "Heatmap of Key Access Frequency by Hour of Day"
PLOT_KEY_USAGE_HEATMAP_X_LABEL = "Key"
PLOT_KEY_USAGE_HEATMAP_Y_LABEL = "Hour of Day"
PLOT_KEY_USAGE_HEATMAP_COLORBAR_LABEL = "Access Count"

PLOT_HIT_MISS_RATES_PAD = 3.0
PLOT_HIT_MISS_RATES_NUM_ROWS = 2
PLOT_HIT_MISS_RATES_NUM_COLS = 1
PLOT_HIT_RATE_SUBPLOT_LINE_STYLE = "--"
PLOT_MISS_RATE_SUBPLOT_LINE_STYLE = "-"
PLOT_HIT_RATE_SUBPLOT_TITLE = "Hit Rate Over Time"
PLOT_MISS_RATE_SUBPLOT_TITLE = "Miss Rate Over Time"
PLOT_HIT_MISS_RATE_SUBPLOT_X_LABEL = "Request"
PLOT_HIT_RATE_SUBPLOT_Y_LABEL = "Hit Rate (%)"
PLOT_MISS_RATE_SUBPLOT_Y_LABEL = "Miss Rate (%)"
PLOT_HIT_MISS_RATES_SUBPLOTS_TITLE_NAME = "title"
PLOT_HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME = "ylabel"
PLOT_HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME = "line_style"
PLOT_HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME = "transform"
PLOT_HIT_MISS_RATES_SUBPLOTS = [
    {
        PLOT_HIT_MISS_RATES_SUBPLOTS_TITLE_NAME: PLOT_HIT_RATE_SUBPLOT_TITLE,
        PLOT_HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME: PLOT_HIT_RATE_SUBPLOT_Y_LABEL,
        PLOT_HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME: PLOT_HIT_RATE_SUBPLOT_LINE_STYLE,
        PLOT_HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME: lambda hit: hit,
    },
    {
        PLOT_HIT_MISS_RATES_SUBPLOTS_TITLE_NAME: PLOT_MISS_RATE_SUBPLOT_TITLE,
        PLOT_HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME: PLOT_MISS_RATE_SUBPLOT_Y_LABEL,
        PLOT_HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME: PLOT_MISS_RATE_SUBPLOT_LINE_STYLE,
        PLOT_HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME: lambda hit: 100 - hit,
    },
]
