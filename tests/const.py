"""const.py

Centralized module defining static constants required for testing.

This module consolidates fixed values, default names, and structural identifiers
used during testing.
"""

from const import DATASET_COLUMN_REQUEST_NAME, DATASET_COLUMN_TIMESTAMP_NAME

# ----------------------------
# Data Testing
# ----------------------------
DATA_TESTING_DATA_SOURCE_NAME_DEFAULT = "data_source"
DATA_TESTING_DATAFRAME_ASSET_NAME_DEFAULT = "dataframe_asset"
DATA_TESTING_BATCH_DEFINITION_NAME_DEFAULT = "batch_definition"
DATA_TESTING_EXPECTATION_SUITE_NAME_DEFAULT = "Dataset expectations"
DATA_TESTING_VALIDATION_DEFINITION_NAME = "Validation definition"
DATA_TESTING_CHECKPOINT_NAME = "Checkpoint"
DATA_TESTING_CHECKPOINT_BATCH_PARAMETERS_DATAFRAME_KEY_NAME = "dataframe"


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

DATASET_COLUMN_SIN_COS_TIME_MIN_VALUE = -1.0
DATASET_COLUMN_SIN_COS_TIME_MAX_VALUE = 1.0
