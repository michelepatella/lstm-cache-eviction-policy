import logging
from datetime import timedelta
from typing import Literal

# Configuration
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PARENT_LEVEL = 5
SECTIONS_WITH_PARAMS = {
    "model": "params",
    "training.optimizer": "params",
}

# Logs
LOGS_FORMAT = "[%(phase)s] %(levelname)s: %(message)s"
LOGS_SAVE_PATH = "./logs/log.log"
LOGS_MAX_BYTES = 10_000_000
LOGS_BACKUP_COUNT = 100
LOGS_DEFAULT_LEVEL = logging.INFO
LOGS_CONFIGURATION_PHASE = "configuration"
LOGS_DATA_GENERATION_PHASE = "data generation"
LOGS_DATA_PREPROCESSING_PHASE = "data preprocessing"
LOGS_VALIDATION_PHASE = "validation"

# Time
MIN_HOUR = 0
MAX_HOUR = 23
PERIOD = timedelta(days=1).total_seconds()
SECONDS_IN_HOUR = timedelta(hours=1).total_seconds()
HOURS_IN_DAY = timedelta(days=1).total_seconds() / SECONDS_IN_HOUR

# Data distribution
DATA_DISTRIBUTION_STATIC_MODE = "static"
DATA_DISTRIBUTION_DYNAMIC_MODE = "dynamic"
ALLOWED_DATA_DISTRIBUTION_MODES = [
    DATA_DISTRIBUTION_STATIC_MODE,
    DATA_DISTRIBUTION_DYNAMIC_MODE,
]

# Dataset Splitting
TRAINING_SPLIT_TYPE = "training"
TESTING_SPLIT_TYPE = "testing"

# Dataset columns
TIMESTAMP_COLUMN_NAME = "timestamp"
REQUEST_COLUMN_NAME = "request"
SIN_TIME_COLUMN_NAME = "sin_time"
COS_TIME_COLUMN_NAME = "cos_time"

# Training
TRAINING_OPTIMIZER_ADAM = "adam"
TRAINING_OPTIMIZER_ADAMW = "adamw"
TRAINING_OPTIMIZER_SGD = "sgd"
ALLOWED_TRAINING_OPTIMIZERS = [
    TRAINING_OPTIMIZER_ADAM,
    TRAINING_OPTIMIZER_ADAMW,
    TRAINING_OPTIMIZER_SGD,
]

# Validation
VALIDATION_PARAMS_SUFFIX = "_range"

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
