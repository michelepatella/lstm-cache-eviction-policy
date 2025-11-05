"""const.py

Centralized module for defining project-wide constants, configuration
settings, and magic strings.

This module organizes all static values, file paths, default parameters,
logging field names, and plot configurations used across various
components of the project.
"""

from pathlib import Path
from typing import Literal

import torch

from const import DATASET_COLUMN_REQUEST_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# API
# ----------------------------
API_ENV_VAR_FULL_URL_NAME = "API_ENDPOINT_FULL_URL"

API_PARAM_KEYS_IN_CACHE_NAME = "keys_in_cache"
API_PARAM_LAST_ACCESSES_NAME = "last_accesses"
API_PARAM_USER_API_KWARGS_NAME = "user_api_kwargs"


# ----------------------------
# Autoregressive Rollout
# ----------------------------
AUTOREGRESSIVE_ROLLOUT_LAST_TIME_IDX = -1
AUTOREGRESSIVE_ROLLOUT_LAST_TIME_BATCH_IDX = 0
AUTOREGRESSIVE_ROLLOUT_TIME_ARRAY_IDX = 0
AUTOREGRESSIVE_ROLLOUT_SEQUENCE_SHIFT_IDX = 1


# ----------------------------
# Data
# ----------------------------
DATA_GENERATION_TIMESTAMPS_START = 0.0
DATA_GENERATION_CURRENT_DAY_START = 0
DATA_GENERATION_CURRENT_SECONDS_IN_DAY_START = 0.0


# ----------------------------
# Dataset
# ----------------------------
DATASET_COLUMN_SIN_TIME_NAME = "sin_time"
DATASET_COLUMN_COS_TIME_NAME = "cos_time"

DATASET_PROCESSED_COLUMNS = [
    DATASET_COLUMN_SIN_TIME_NAME,
    DATASET_COLUMN_COS_TIME_NAME,
    DATASET_COLUMN_REQUEST_NAME,
]

DATASET_PROCESSED_FEATURE_COLUMNS = [
    DATASET_COLUMN_SIN_TIME_NAME,
    DATASET_COLUMN_COS_TIME_NAME,
]

DATASET_INDEX = False

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
# Early Stopping
# ----------------------------
EARLY_STOPPING_TRIGGERED = True
EARLY_STOPPING_UNTRIGGERED = False


# ----------------------------
# Grid Search
# ----------------------------
GRID_SEARCH_DESC = "Grid Search"


# ----------------------------
# JSON
# ----------------------------
JSON_WRAP_BOX = True
JSON_INDENT = 4


# ----------------------------
# Logs
# ----------------------------
LOGS_FIELD_STANDARD_NAMES = {
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

LOGS_FIELD_PHASE_NAME = "phase"
LOGS_FIELD_PHASE_DEFAULT = "unknown"
LOGS_FIELD_TIMESTAMP_NAME = "timestamp"
LOGS_FIELD_LEVEL_NAME = "level"
LOGS_FIELD_MESSAGE_NAME = "message"

LOGS_ENV_VAR_ELASTIC_ENDPOINT_NAME = "ELASTIC_ENDPOINT"
LOGS_ENV_VAR_ELASTIC_TOKEN_NAME = "ELASTIC_TOKEN"
LOGS_ENV_VAR_ELASTIC_INDEX_NAME = "ELASTIC_INDEX"

LOGS_ACTIONS_FIELD_INDEX_NAME = "_index"
LOGS_ACTIONS_FIELD_SOURCE_NAME = "_source"

LOGS_THREAD_DAEMON = True


# ----------------------------
# Missing Values Removal
# ----------------------------
MISSING_VALUES_REMOVAL_DROPNA_AXIS = 0


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

MODEL_TRAIN_MODE = "train"
MODEL_EVAL_MODE = "eval"
MODEL_MC_DROPOUT_MODE = "mc_dropout"

MODEL_TRAINED_STATIC_FILE_PATH = (
    PROJECT_ROOT / "models" / "static" / "trained_static_model.pt"
)
MODEL_TRAINED_DYNAMIC_FILE_PATH = (
    PROJECT_ROOT / "models" / "dynamic" / "trained_dynamic_model.pt"
)

MODEL_COMPUTE_METRICS_DEFAULT = False

MODEL_METRICS_AVG_LOSS_NAME = "avg_loss"
MODEL_METRICS_CLASS_REPORT_NAME = "class_report"
MODEL_METRICS_COHEN_KAPPA_SCORE_NAME = "cohen_kappa_score"
MODEL_METRICS_ACCURACY_NAME = "accuracy"
MODEL_METRICS_MACRO_AVG_NAME = "macro avg"
MODEL_METRICS_WEIGHTED_AVG_NAME = "weighted avg"
MODEL_METRICS_CLASS_REPORT_OUTPUT_DICT = True
MODEL_METRICS_CLASS_REPORT_ZERO_DIVISION = 0

MODEL_LOADING_WEIGHTS_ONLY = False

MODEL_OPTIMIZATION_PRUNING_PARAMS = ("weight",)


# ----------------------------
# Monte Carlo (MC) Dropout
# ----------------------------
MC_DROPOUT_FLAG_NAME = "mc_dropout"

MC_DROPOUT_DISABLED = False
MC_DROPOUT_ENABLED = True

MC_DROPOUT_NUM_SAMPLES_DEFAULT = 1


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


# ----------------------------
# Simulations Metrics
# ----------------------------
SIMULATIONS_METRICS_TIMELINE_INDEX_NAME = "index"
SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME = "instant_hit_rate"


# ----------------------------
# Tensor
# ----------------------------
TENSOR_OUTPUTS_BATCH_DIM = 0
TENSOR_FEATURES_DIM = 1
TENSOR_TARGET_DIM = 1
TENSOR_SEQUENCE_DIM = -1


# ----------------------------
# Time
# ----------------------------
TIME_HOURS_IN_DAY = 24
TIME_SECONDS_IN_DAY = 86_400
TIME_SECONDS_IN_HOUR = 3_600
TIME_MICROSECONDS_IN_SECOND = 1_000_000


# ----------------------------
# Torch
# ----------------------------
TORCH_DTYPE = torch.float32


# ----------------------------
# Training
# ----------------------------
TRAINING_EPOCHS_DESC = "Training"
TRAINING_SINGLE_EPOCH_DESC = "Epoch"


# ----------------------------
# YAML
# ----------------------------
YAML_DUMP_SORT_KEYS = False
