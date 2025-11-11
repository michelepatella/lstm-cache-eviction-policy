"""test_processed_data_integrity.py

Module containing the test function for verifying the integrity
of the processed data using the Deepchecks library.

This module ensures that the data, after processing, meets configured
data integrity constraints.

Functions:
    test_processed_data_integrity() -> None
        Runs a Deepchecks Suite on the processed data based on configurations.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    IsSingleValue,
    PercentOfNulls,
)

from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.const import (
    DATA_INTEGRITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH,
    DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME,
)
from tests.data.integrity.helpers import (
    initialize_data_integrity_tests,
    run_data_integrity_tests,
)


def test_processed_data_integrity() -> None:
    """Runs the Deepchecks test suite against the processed data.

    This test verifies the integrity of the processed data by ensuring it adheres
    to predefined configurations for integrity checks, including:
        - PercentOfNulls
        - FeatureLabelCorrelation
        - FeatureFeatureCorrelation
        - IsSingleValue

    Returns:
        None
    """
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()

    # Initialization
    dc_dataset, tests_config = initialize_data_integrity_tests(
        DATASET_PROCESSED_TYPE,
        pipeline_config.data.general.mode,
    )

    # Build a suite of checks
    suite = Suite(
        DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME,
        PercentOfNulls(
            random_state=tests_config.seed.value,
            n_samples=pipeline_config.data.general.requests,
        ).add_condition_percent_of_nulls_not_greater_than(
            tests_config.data.processed.integrity.percent_of_nulls.threshold,
        ),
        FeatureLabelCorrelation(
            random_state=tests_config.seed.value,
            n_samples=pipeline_config.data.general.requests,
        ).add_condition_feature_pps_less_than(
            tests_config.data.processed.integrity.feature_label_correlation.threshold,
        ),
        FeatureFeatureCorrelation(
            random_state=tests_config.seed.value,
            n_samples=pipeline_config.data.general.requests,
        ).add_condition_max_number_of_pairs_above_threshold(
            tests_config.data.processed.integrity.feature_feature_correlation.threshold,
            tests_config.data.processed.integrity.feature_feature_correlation.num_pairs,
        ),
        IsSingleValue(
            random_state=tests_config.seed.value,
            n_samples=pipeline_config.data.general.requests,
        ).add_condition_not_single_value(),
    )

    # Run all integrity tests
    run_data_integrity_tests(
        dc_dataset,
        suite,
        str(DATA_INTEGRITY_TESTS_PROCESSED_DATA_RESULTS_SAVE_PATH),
    )


if __name__ == "__main__":
    test_processed_data_integrity()
