"""test_processed_data_quality.py

Module dedicated to running data quality validation checks on the processed
dataset using the Great Expectations (GX) framework.

This module loads the processed dataset and applies a comprehensive
Expectation Suite to verify its quality. The checks specifically focus on:
    - Completeness: Ensures no critical columns contain null values.
    - Numeric Constraints: Validates that engineered features and request
                           keys are within their expected numerical bounds.
    - Schema Quality: Verifies the existence, data types, count, and order
                        of the processed columns.

Functions:
    test_processed_data_quality() -> None
        Executes the full suite of data quality expectations for the processed data.
"""

from tests.helpers.data_quality_helpers import (
    add_column_count_expectation,
    add_column_existence_expectations,
    add_column_not_null_expectations,
    add_column_order_expectation,
    add_column_range_expectations,
    add_column_type_expectations,
    initialize_data_quality_tests,
    run_data_quality_tests,
)

from components.const import (
    DATASET_COLUMN_COS_TIME_NAME,
    DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    DATASET_COLUMN_LOCAL_RECENCY_NAME,
    DATASET_COLUMN_SIN_TIME_NAME,
    DATASET_PROCESSED_COLUMNS,
)
from const import DATASET_COLUMN_REQUEST_NAME
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.const import (
    DATA_QUALITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH,
    DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
    DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
    DATASET_COLUMN_LOCAL_FREQUENCY_TYPE,
    DATASET_COLUMN_LOCAL_RECENCY_TYPE,
    DATASET_COLUMN_REQUEST_TYPE,
    DATASET_COLUMN_SIN_COS_TIME_MAX_VALUE,
    DATASET_COLUMN_SIN_COS_TIME_MIN_VALUE,
    DATASET_COLUMN_SIN_COS_TIME_TYPE,
)


def test_processed_data_quality() -> None:
    """Tests the quality and schema of the processed data.

    This function performs a complete data validation check on the data
    after preprocessing by:
        - Initializing the Great Expectations environment.
        - Defining Completeness Expectations (non-null checks).
        - Defining Numeric Expectations (range checks for features and request keys).
        - Defining Schema Expectations (existence, type, count, and order of columns).
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
            DATASET_PROCESSED_TYPE,
            pipeline_config.data.general.mode,
        )
    )

    # ----------------------------
    # Completeness checks
    # ----------------------------
    # Ensure no columns have NaN as value
    add_column_not_null_expectations(suite, DATASET_PROCESSED_COLUMNS)

    # ----------------------------
    # Numeric checks
    # ----------------------------
    # Ensure features are within the valid range,
    # as well as requests are between min and max keys
    add_column_range_expectations(
        suite,
        {
            DATASET_COLUMN_SIN_TIME_NAME: (
                DATASET_COLUMN_SIN_COS_TIME_MIN_VALUE,
                DATASET_COLUMN_SIN_COS_TIME_MAX_VALUE,
            ),
            DATASET_COLUMN_COS_TIME_NAME: (
                DATASET_COLUMN_SIN_COS_TIME_MIN_VALUE,
                DATASET_COLUMN_SIN_COS_TIME_MAX_VALUE,
            ),
            DATASET_COLUMN_LOCAL_FREQUENCY_NAME: (
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
            ),
            DATASET_COLUMN_LOCAL_RECENCY_NAME: (
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
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
            DATASET_COLUMN_SIN_TIME_NAME: DATASET_PROCESSED_COLUMNS.index(
                DATASET_COLUMN_SIN_TIME_NAME,
            ),
            DATASET_COLUMN_COS_TIME_NAME: DATASET_PROCESSED_COLUMNS.index(
                DATASET_COLUMN_COS_TIME_NAME,
            ),
            DATASET_COLUMN_LOCAL_FREQUENCY_NAME: DATASET_PROCESSED_COLUMNS.index(
                DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
            ),
            DATASET_COLUMN_LOCAL_RECENCY_NAME: DATASET_PROCESSED_COLUMNS.index(
                DATASET_COLUMN_LOCAL_RECENCY_NAME,
            ),
            DATASET_COLUMN_REQUEST_NAME: DATASET_PROCESSED_COLUMNS.index(
                DATASET_COLUMN_REQUEST_NAME,
            ),
        },
    )

    # Ensure columns are of correct type
    add_column_type_expectations(
        suite,
        {
            DATASET_COLUMN_SIN_TIME_NAME: DATASET_COLUMN_SIN_COS_TIME_TYPE,
            DATASET_COLUMN_COS_TIME_NAME: DATASET_COLUMN_SIN_COS_TIME_TYPE,
            DATASET_COLUMN_LOCAL_FREQUENCY_NAME: DATASET_COLUMN_LOCAL_FREQUENCY_TYPE,
            DATASET_COLUMN_LOCAL_RECENCY_NAME: DATASET_COLUMN_LOCAL_RECENCY_TYPE,
            DATASET_COLUMN_REQUEST_NAME: DATASET_COLUMN_REQUEST_TYPE,
        },
    )

    # Ensure the number of columns is correct
    add_column_count_expectation(suite, len(DATASET_PROCESSED_COLUMNS))

    # Ensure columns are sorted as expected
    add_column_order_expectation(suite, DATASET_PROCESSED_COLUMNS)

    # ----------------------------
    # Suite running
    # ----------------------------
    run_data_quality_tests(
        context,
        batch_definition,
        suite,
        df,
        DATA_QUALITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH,
    )


if __name__ == "__main__":
    test_processed_data_quality()
