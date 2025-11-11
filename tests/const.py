"""const.py

Centralized module defining static constants required for testing.

This module consolidates fixed values, default names, and structural identifiers
used during testing.
"""

from pathlib import Path

from const import DATASET_COLUMN_REQUEST_NAME, DATASET_COLUMN_TIMESTAMP_NAME

PROJECT_ROOT = Path(__file__).parent.parent


# ----------------------------
# Data Integrity Tests
# ----------------------------
DATA_INTEGRITY_TESTS_RAW_DATA_SUITE_NAME = "Raw data integry tests"
DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME = (
    "Processed data integrity tests"
)

DATA_INTEGRITY_TESTS_RAW_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "data"
    / "integrity"
    / "raw_data_results.html"
)
DATA_INTEGRITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "data"
    / "integrity"
    / "processed_data_results.html"
)

DATA_INTEGRITY_RESULTS_HTML_AS_WIDGET = False
DATA_INTEGRITY_RESULTS_HTML_REQUIREJS = False


# ----------------------------
# Data Quality Tests
# ----------------------------
DATA_QUALITY_TESTS_DATA_SOURCE_NAME_DEFAULT = "data_source"
DATA_QUALITY_TESTS_DATAFRAME_ASSET_NAME_DEFAULT = "dataframe_asset"
DATA_QUALITY_TESTS_BATCH_DEFINITION_NAME_DEFAULT = "batch_definition"
DATA_QUALITY_TESTS_EXPECTATION_SUITE_NAME_DEFAULT = "Dataset expectations"
DATA_QUALITY_TESTS_VALIDATION_DEFINITION_NAME_DEFAULT = "Validation definition"
DATA_QUALITY_TESTS_CHECKPOINT_NAME_DEFAULT = "Checkpoint"

DATA_QUALITY_TESTS_CHECKPOINT_BATCH_PARAMETERS_DATAFRAME_KEY_NAME = "dataframe"

DATA_QUALITY_TESTS_RAW_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "data"
    / "quality"
    / "raw_data_results.json"
)
DATA_QUALITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "data"
    / "quality"
    / "processed_data_results.json"
)


# ----------------------------
# Dataset
# ----------------------------
DATASET_RAW_COLUMNS = [
    DATASET_COLUMN_TIMESTAMP_NAME,
    DATASET_COLUMN_REQUEST_NAME,
]

DATASET_COLUMN_TIMESTAMP_TYPE = "float"
DATASET_COLUMN_REQUEST_TYPE = "int"
DATASET_COLUMN_SIN_COS_TIME_TYPE = "float"
DATASET_COLUMN_LOCAL_FREQUENCY_TYPE = "float"
DATASET_COLUMN_LOCAL_RECENCY_TYPE = "float"

DATASET_COLUMN_SIN_COS_TIME_MIN_VALUE = -1.0
DATASET_COLUMN_SIN_COS_TIME_MAX_VALUE = 1.0
DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE = 0.0
DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE = 1.0


# ----------------------------
# Tests Config
# ----------------------------
TESTS_CONFIG_FILE_PATH = (
    PROJECT_ROOT / "tests" / "config" / "tests_config.yaml"
)
