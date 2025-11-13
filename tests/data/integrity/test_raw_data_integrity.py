"""test_raw_data_integrity.py

Module containing the test function for verifying the integrity
of the raw data using the Deepchecks library.

This module ensures that the raw, unprocessed data meets configured
data integrity constraints.

Functions:
    test_raw_data_integrity() -> None
        Runs a Deepchecks Suite on the raw data based on configurations.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    IsSingleValue,
    PercentOfNulls,
)

from const import DATASET_RAW_TYPE
from pipeline.config.configurator import prepare_pipeline_config
from tests.const import (
    DATA_INTEGRITY_TESTS_RAW_DATA_RESULTS_SAVE_PATH,
    DATA_INTEGRITY_TESTS_RAW_DATA_SUITE_NAME,
)
from tests.helpers.data_integrity_helpers import (
    initialize_data_integrity_tests,
)
from tests.helpers.dc_helpers import run_dc_suite


def test_raw_data_integrity() -> None:
    """Runs the Deepchecks test suite against the raw data.

    This test verifies the integrity of the raw data by ensuring it adheres
    to predefined configurations for integrity checks, including:
        - PercentOfNulls
        - FeatureLabelCorrelation
        - FeatureFeatureCorrelation
        - IsSingleValue

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()

    # Initialization
    dc_dataset, tests_config = initialize_data_integrity_tests(
        DATASET_RAW_TYPE,
        pipeline_config.data.general.mode,
    )

    # ----------------------------
    # Suite building
    # ----------------------------
    suite = Suite(
        DATA_INTEGRITY_TESTS_RAW_DATA_SUITE_NAME,
        PercentOfNulls(
            random_state=tests_config.seed.value,
        ).add_condition_percent_of_nulls_not_greater_than(
            threshold=tests_config.data.raw.integrity.percent_of_nulls.threshold,
        ),
        FeatureLabelCorrelation(
            random_state=tests_config.seed.value,
        ).add_condition_feature_pps_less_than(
            threshold=tests_config.data.raw.integrity.feature_label_correlation.threshold,
        ),
        FeatureFeatureCorrelation(
            random_state=tests_config.seed.value,
        ).add_condition_max_number_of_pairs_above_threshold(
            threshold=tests_config.data.raw.integrity.feature_feature_correlation.threshold,
            n_pairs=tests_config.data.raw.integrity.feature_feature_correlation.num_pairs,
        ),
        IsSingleValue(
            random_state=tests_config.seed.value,
        ).add_condition_not_single_value(),
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    run_dc_suite(
        dc_dataset,
        suite,
        str(DATA_INTEGRITY_TESTS_RAW_DATA_RESULTS_SAVE_PATH),
    )


if __name__ == "__main__":
    test_raw_data_integrity()
