import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal

# Project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration
CONFIG_FILE_NAME = "config.yaml"
CONFIG_SECTIONS_WITH_PARAMS = {
    "model": "params",
    "training.optimizer": "params",
}

# Logs
LOGS_FORMAT = "[%(phase)s] %(levelname)s: %(message)s"
LOGS_SAVE_PATH = PROJECT_ROOT / "logs" / "log.log"
LOGS_MAX_BYTES = 10_000_000
LOGS_BACKUP_COUNT = 100
LOGS_DEFAULT_LEVEL = logging.INFO
LOGS_CONFIGURATION_PHASE = "configuration"
LOGS_DATA_GENERATION_PHASE = "data generation"
LOGS_DATA_PREPROCESSING_PHASE = "data preprocessing"
LOGS_VALIDATION_PHASE = "validation"
LOGS_TRAINING_PHASE = "training"
LOGS_TESTING_PHASE = "testing"

# Time
MIN_HOUR = 0
MAX_HOUR = 23
PERIOD = timedelta(days=1).total_seconds()
SECONDS_IN_HOUR = timedelta(hours=1).total_seconds()
HOURS_IN_DAY = timedelta(days=1).total_seconds() / SECONDS_IN_HOUR

# Hardware
DEVICES = ["cpu", "cuda", "mps"]

# Data distribution
DATA_DISTRIBUTION_STATIC_MODE = "static"
DATA_DISTRIBUTION_DYNAMIC_MODE = "dynamic"
DATA_DISTRIBUTION_MODES = [
    DATA_DISTRIBUTION_STATIC_MODE,
    DATA_DISTRIBUTION_DYNAMIC_MODE,
]

# Dataset
DATASET_RAW_TYPE = "raw"
DATASET_PREPROCESSED_TYPE = "preprocessed"
TIMESTAMP_COLUMN = "timestamp"
REQUEST_COLUMN = "request"
SIN_TIME_COLUMN = "sin_time"
COS_TIME_COLUMN = "cos_time"
TRAINING_SPLIT_TYPE = "training"
TESTING_SPLIT_TYPE = "testing"

# Data preprocessing
MISSING_VALUES_REMOVAL_DROPNA_AXIS = 0
MISSING_VALUES_REMOVAL_DROPNA_HOW: Literal["any", "all"] = "any"

# Training
ADAM_OPTIMIZER = "adam"
ADAMW_OPTIMIZER = "adamw"
SGD_OPTIMIZER = "sgd"

TRAINING_OPTIMIZERS = [
    ADAM_OPTIMIZER,
    ADAMW_OPTIMIZER,
    SGD_OPTIMIZER,
]

# Validation
VALIDATION_PARAMS_SUFFIX = "_range"

# Inference
MC_DROPOUT_SAMPLES_DEFAULT = 1

# Model metrics
COMPUTE_METRICS_DEFAULT = False
MODEL_METRICS_CLASS_REPORT_NAME = "class_report"
MODEL_METRICS_TOP_K_ACCURACY = "top_k_accuracy"
MODEL_METRICS_CONFUSION_MATRIX = "confusion_matrix"
MODEL_METRICS_COHEN_KAPPA_SCORE = "cohen_kappa_score"

# Class weight
CLASS_WEIGHT_TYPE = "balanced"

# Model
LSTM_PARAMETERS = [
    "hidden_size",
    "num_layers",
    "bias",
    "batch_first",
    "dropout",
    "bidirectional",
    "proj_size",
]
MC_DROPOUT_DEFAULT = False

# Figures
FIGURE_SIZE = 12
FIGURE_TITLE_FONT_SIZE = 18
FIGURE_LABEL_FONT_SIZE = 16

# Zipf log-log plot
ZIPF_LOGLOG_PLOT_MARKER = "o"
ZIPF_LOGLOG_PLOT_TITLE = "Zipf Distribution (Log-Log)"
ZIPF_LOGLOG_PLOT_X_LABEL = "Key"
ZIPF_LOGLOG_PLOT_Y_LABEL = "Frequency"

# Daily profile plot
DAILY_PROFILE_PLOT_BIN_SIZE = 0.5
DAILY_PROFILE_PLOT_ALIGN: Literal["center", "edge"] = "center"
DAILY_PROFILE_PLOT_EDGE_COLOR = "black"
DAILY_PROFILE_PLOT_TITLE = "Distribution of Requests Over Day"
DAILY_PROFILE_PLOT_X_LABEL = "Hour of Day"
DAILY_PROFILE_PLOT_Y_LABEL = "Number of Requests"
DAILY_PROFILE_PLOT_STEP = 1

# Key usage heatmap
KEY_USAGE_HEATMAP_ASPECT: Literal["equal", "auto"] = "auto"
KEY_USAGE_HEATMAP_CMAP = "turbo"
KEY_USAGE_HEATMAP_TITLE = "Heatmap of Key Access Frequency by Hour of Day"
KEY_USAGE_HEATMAP_COLORBAR_LABEL = "Access Count"
KEY_USAGE_HEATMAP_X_LABEL = "Key"
KEY_USAGE_HEATMAP_Y_LABEL = "Hour of Day"
KEY_USAGE_HEATMAP_ROTATION = 45

# Precision-recall curve
PRECISION_RECALL_CURVE_TITLE = "Precision-Recall Curve"
PRECISION_RECALL_CURVE_CLASS_LABEL = "Class"
PRECISION_RECALL_CURVE_X_LABEL = "Recall"
PRECISION_RECALL_CURVE_Y_LABEL = "Precision"

# Confusion matrix plot
CONFUSION_MATRIX_PLOT_TITLE = "Confusion Matrix"
CONFUSION_MATRIX_PLOT_X_LABEL = "Predicted Key"
CONFUSION_MATRIX_PLOT_Y_LABEL = "True Key"
CONFUSION_MATRIX_PLOT_ANNOT = True
CONFUSION_MATRIX_PLOT_FMT = "d"
