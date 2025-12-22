"""const.py

Centralized module defining static constants required for testing.

This module consolidates fixed values, default names, and structural identifiers
used during testing.
"""

from pathlib import Path

from components.const import (
    DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    DATASET_COLUMN_LOCAL_RECENCY_NAME,
)
from const import (
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
)

PROJECT_ROOT = Path(__file__).parent.parent


# -----------------------------------
# Data Integrity Tests
# -----------------------------------
DATA_INTEGRITY_TESTS_RAW_DATA_SUITE_NAME = "Raw Data Integrity Tests"
DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME = (
    "Processed Data Integrity Tests"
)

DATA_INTEGRITY_TESTS_RAW_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "integrity"
    / "raw_static_data_integrity_tests_results.html"
)
DATA_INTEGRITY_TESTS_PROCESSED_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "integrity"
    / "processed_static_data_integrity_tests_results.html"
)

DATA_INTEGRITY_TESTS_RAW_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "integrity"
    / "raw_dynamic_data_integrity_tests_results.html"
)
DATA_INTEGRITY_TESTS_PROCESSED_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "integrity"
    / "processed_dynamic_data_integrity_tests_results.html"
)

DATA_INTEGRITY_TESTS_RAW_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "integrity"
    / "raw_real_data_integrity_tests_results.html"
)
DATA_INTEGRITY_TESTS_PROCESSED_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "integrity"
    / "processed_real_data_integrity_tests_results.html"
)


# -----------------------------------
# Data Quality Tests
# -----------------------------------
DATA_QUALITY_TESTS_DATA_SOURCE_NAME_DEFAULT = "data_source"
DATA_QUALITY_TESTS_DATAFRAME_ASSET_NAME_DEFAULT = "dataframe_asset"
DATA_QUALITY_TESTS_BATCH_DEFINITION_NAME_DEFAULT = "batch_definition"
DATA_QUALITY_TESTS_EXPECTATION_SUITE_NAME_DEFAULT = "Dataset expectations"
DATA_QUALITY_TESTS_VALIDATION_DEFINITION_NAME_DEFAULT = "Validation definition"
DATA_QUALITY_TESTS_CHECKPOINT_NAME_DEFAULT = "Checkpoint"

DATA_QUALITY_TESTS_CHECKPOINT_BATCH_PARAMETERS_DATAFRAME_KEY_NAME = "dataframe"

DATA_QUALITY_TESTS_RAW_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "quality"
    / "raw_static_data_quality_tests_results.json"
)
DATA_QUALITY_TESTS_PROCESSED_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "quality"
    / "processed_static_data_quality_tests_results.json"
)

DATA_QUALITY_TESTS_RAW_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "quality"
    / "raw_dynamic_data_quality_tests_results.json"
)
DATA_QUALITY_TESTS_PROCESSED_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "quality"
    / "processed_dynamic_data_quality_tests_results.json"
)

DATA_QUALITY_TESTS_RAW_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "quality"
    / "raw_real_data_quality_tests_results.json"
)
DATA_QUALITY_TESTS_PROCESSED_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "quality"
    / "processed_real_data_quality_tests_results.json"
)


# -----------------------------------
# Data Train Test Validation Tests
# -----------------------------------
DATA_TRAIN_TEST_VALIDATION_TESTS_SUITE_NAME = (
    "Data Train Test Validation Tests"
)

DATA_TRAIN_VALIDATION_TESTS_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "train_test_validation"
    / "static_data_train_validation_tests_results.html"
)
DATA_TRAIN_TEST_TESTS_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "data"
    / "train_test_validation"
    / "static_data_train_test_tests_results.html"
)

DATA_TRAIN_VALIDATION_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "train_test_validation"
    / "dynamic_data_train_validation_tests_results.html"
)
DATA_TRAIN_TEST_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "data"
    / "train_test_validation"
    / "dynamic_data_train_test_tests_results.html"
)

DATA_TRAIN_VALIDATION_TESTS_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "train_test_validation"
    / "real_data_train_validation_tests_results.html"
)
DATA_TRAIN_TEST_TESTS_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "data"
    / "train_test_validation"
    / "real_data_train_test_tests_results.html"
)

DATA_TRAIN_TEST_VALIDATION_TESTS_ADD_INDEX_COLUMN = True
DATA_TRAIN_TEST_VALIDATION_TESTS_REMOVE_SEQ_LEN = False


# -----------------------------------
# Dataset
# -----------------------------------
DATASET_COLUMN_TEMP_INDEX_NAME = "_index"

DATASET_RAW_COLUMNS = [
    DATASET_COLUMN_TIMESTAMP_NAME,
    DATASET_COLUMN_REQUEST_NAME,
]

DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS = [
    DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    DATASET_COLUMN_LOCAL_RECENCY_NAME,
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


# -----------------------------------
# DeepChecks
# -----------------------------------
DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET = False
DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS = False

DEEP_CHECKS_LABEL_TYPE = "multiclass"


# -----------------------------------
# L1
# -----------------------------------
L1_MAX_DISTANCE = 2.0


# -----------------------------------
# Model Behavioral Directional Tests
# -----------------------------------
MODEL_BEHAVIORAL_DIRECTIONAL_TESTS_LOCAL_FEATURE_PERTURBATIONS_PARAM_FEATURE_NAME = "feature"


# -----------------------------------
# Model Behavioral Invariance Tests
# -----------------------------------
MODEL_BEHAVIORAL_INVARIANCE_TESTS_FEATURE_PERTURBATIONS_PARAM_FEATURE_IDX_NAME = "feature_idx"


# -----------------------------------
# Model Performance Tests
# -----------------------------------
MODEL_PERFORMANCE_TESTS_SUITE_NAME = "Model Performance Tests"

MODEL_PERFORMANCE_TESTS_STATIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "static"
    / "model"
    / "performance"
    / "static_model_performance_tests_results.html"
)
MODEL_PERFORMANCE_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "dynamic"
    / "model"
    / "performance"
    / "dynamic_model_performance_tests_results.html"
)
MODEL_PERFORMANCE_TESTS_REAL_DATA_RESULTS_SAVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tests"
    / "real"
    / "model"
    / "performance"
    / "real_model_performance_tests_results.html"
)

MODEL_PERFORMANCE_TESTS_ADD_INDEX_COLUMN = False
MODEL_PERFORMANCE_TESTS_REMOVE_SEQ_LEN = True


# -----------------------------------
# Model Training Tests
# -----------------------------------
MODEL_TRAINING_TESTS_PORTABILITY_TEST_PARAM_DEVICE_TYPE_NAME = "device_type"


# -----------------------------------
# Tests Config
# -----------------------------------
TESTS_CONFIG_FILE_PATH = (
    PROJECT_ROOT / "tests" / "config" / "tests_config.yaml"
)


# -----------------------------------
# Torch
# -----------------------------------
TORCH_TOP_K_LARGEST = True
TORCH_TOP_K_SMALLEST = False
