"""test_raw_data_quality.py

Module dedicated to running data quality validation checks on the initial,
unprocessed (raw) data using the Great Expectations (GX) framework.

This module loads the raw data and applies a comprehensive Expectation Suite
to verify its quality before any pipeline transformations occur. The checks
cover:
    - Numeric Constraints: Ensures timestamp and request values fall within
                           defined boundaries.
    - Schema Quality: Verifies the existence, data types, count, and order
                      of the essential columns.
    - Volume Check: Asserts that the dataset contains the expected number
                    of rows (requests).

Functions:
    test_raw_data_quality() -> None
        Executes the full suite of data quality expectations for the raw data.
"""

import pytest

from const import (
    DATA_DYNAMIC_MODE,
    DATA_STATIC_MODE,
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
    DATASET_RAW_TYPE,
    TIME_END_HOUR,
    TIME_START_HOUR,
)
from pipeline.config.configurator import prepare_pipeline_config
from tests.const import (
    DATA_QUALITY_TESTS_RAW_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    DATA_QUALITY_TESTS_RAW_REAL_DATA_RESULTS_SAVE_PATH,
    DATA_QUALITY_TESTS_RAW_STATIC_DATA_RESULTS_SAVE_PATH,
    DATASET_COLUMN_REQUEST_TYPE,
    DATASET_COLUMN_TIMESTAMP_TYPE,
    DATASET_RAW_COLUMNS,
)
from tests.data.quality.helpers import (
    add_column_count_expectation,
    add_column_existence_expectations,
    add_column_order_expectation,
    add_column_range_expectations,
    add_column_type_expectations,
    add_row_count_expectation,
    initialize_data_quality_tests,
    run_data_quality_tests,
)


@pytest.mark.slow
@pytest.mark.data_quality_raw
@pytest.mark.after_data_preparation
def test_raw_data_quality() -> None:
    """Tests the quality and schema of the raw data.

    This function performs a complete data validation check on the raw data
    by:
        - Initializing the Great Expectations environment.
        - Defining Numeric Expectations (range checks for timestamps and requests).
        - Defining Schema Expectations (existence, type, count, and order of columns).
        - Defining Volume Expectations (total row count).
        - Executing the Checkpoint and asserting that all expectations are met.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    pipeline_config = prepare_pipeline_config()
    df, context, data_source, data_asset, batch_definition, suite = (
        initialize_data_quality_tests(
            DATASET_RAW_TYPE,
            pipeline_config.data.general.mode,
        )
    )

    # ----------------------------
    # Numeric checks
    # ----------------------------
    # Ensure timestamps (in hours) are
    # within the valid range, as well as requests
    # are between min and max keys
    add_column_range_expectations(
        suite,
        {
            DATASET_COLUMN_TIMESTAMP_NAME: (
                TIME_START_HOUR,
                TIME_END_HOUR + 1,
            ),
            DATASET_COLUMN_REQUEST_NAME: (
                pipeline_config.data.general.keys.min,
                pipeline_config.data.general.keys.max,
            ),
        },
    )

    # ----------------------------
    # Schema checks
    # ----------------------------
    # Ensure columns existence
    add_column_existence_expectations(
        suite,
        {
            DATASET_COLUMN_TIMESTAMP_NAME: DATASET_RAW_COLUMNS.index(
                DATASET_COLUMN_TIMESTAMP_NAME,
            ),
            DATASET_COLUMN_REQUEST_NAME: DATASET_RAW_COLUMNS.index(
                DATASET_COLUMN_REQUEST_NAME,
            ),
        },
    )

    # Ensure columns are of correct type
    add_column_type_expectations(
        suite,
        {
            DATASET_COLUMN_TIMESTAMP_NAME: DATASET_COLUMN_TIMESTAMP_TYPE,
            DATASET_COLUMN_REQUEST_NAME: DATASET_COLUMN_REQUEST_TYPE,
        },
    )

    # Ensure the number of columns is correct
    add_column_count_expectation(suite, len(DATASET_RAW_COLUMNS))

    # Ensure columns are sorted as expected
    add_column_order_expectation(suite, DATASET_RAW_COLUMNS)

    # ----------------------------
    # Volume checks
    # ----------------------------
    # Ensure dataset has expected volume
    add_row_count_expectation(suite, pipeline_config.data.general.requests)

    # ----------------------------
    # Suite running
    # ----------------------------
    # Determine the save path based on
    # the data mode
    if pipeline_config.data.general.mode == DATA_STATIC_MODE:
        save_path = DATA_QUALITY_TESTS_RAW_STATIC_DATA_RESULTS_SAVE_PATH
    elif pipeline_config.data.general.mode == DATA_DYNAMIC_MODE:
        save_path = DATA_QUALITY_TESTS_RAW_DYNAMIC_DATA_RESULTS_SAVE_PATH
    else:
        save_path = DATA_QUALITY_TESTS_RAW_REAL_DATA_RESULTS_SAVE_PATH

    # Run tests suite
    run_data_quality_tests(
        context,
        batch_definition,
        suite,
        df,
        save_path,
    )


if __name__ == "__main__":
    test_raw_data_quality()
