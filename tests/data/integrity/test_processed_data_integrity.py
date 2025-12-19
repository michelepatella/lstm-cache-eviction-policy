"""test_processed_data_integrity.py

Module containing the test function for verifying the integrity
of the processed data using the Deepchecks library.

This module ensures that the data, after processing, meets configured
data integrity constraints.

Functions:
    test_processed_data_integrity() -> None
        Runs a Deepchecks Suite on the processed data based on configurations.
"""

import pytest
from deepchecks import Suite
from deepchecks.tabular.checks import (
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    IsSingleValue,
    PercentOfNulls,
)

from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE
from src.const import DATA_DYNAMIC_MODE, DATA_STATIC_MODE
from tests.const import (
    DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME,
    DATA_INTEGRITY_TESTS_PROCESSED_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    DATA_INTEGRITY_TESTS_PROCESSED_REAL_DATA_RESULTS_SAVE_PATH,
    DATA_INTEGRITY_TESTS_PROCESSED_STATIC_DATA_RESULTS_SAVE_PATH,
)
from tests.data.integrity.helpers import (
    initialize_data_integrity_tests,
)
from tests.helpers.dc_helpers import run_dc_suite


@pytest.mark.data_integrity_processed
@pytest.mark.after_data_processing
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
    # ----------------------------
    # Setup
    # ----------------------------
    # Prepare configuration
    pipeline_config = prepare_pipeline_config()

    # Initialization
    dc_dataset, tests_config = initialize_data_integrity_tests(
        DATASET_PROCESSED_TYPE,
        pipeline_config.data.general.mode,
    )

    # ----------------------------
    # Suite building
    # ----------------------------
    suite = Suite(
        DATA_INTEGRITY_TESTS_PROCESSED_DATA_SUITE_NAME,
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(
            threshold=tests_config.data.integrity.processed.nulls_perc.max,
        ),
        FeatureLabelCorrelation().add_condition_feature_pps_less_than(
            threshold=tests_config.data.integrity.processed.feat_label_corr.max,
        ),
        FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
            threshold=tests_config.data.integrity.processed.feat_feat_corr.max,
            n_pairs=tests_config.data.integrity.processed.feat_feat_corr.max_exceeding_pairs,
        ),
        IsSingleValue().add_condition_not_single_value(),
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Determine the save path based on
    # the data mode
    if pipeline_config.data.general.mode == DATA_STATIC_MODE:
        save_path = (
            DATA_INTEGRITY_TESTS_PROCESSED_STATIC_DATA_RESULTS_SAVE_PATH
        )
    elif pipeline_config.data.general.mode == DATA_DYNAMIC_MODE:
        save_path = (
            DATA_INTEGRITY_TESTS_PROCESSED_DYNAMIC_DATA_RESULTS_SAVE_PATH
        )
    else:
        save_path = DATA_INTEGRITY_TESTS_PROCESSED_REAL_DATA_RESULTS_SAVE_PATH

    # Run tests suite
    run_dc_suite(
        dc_dataset,
        suite,
        str(save_path),
    )


if __name__ == "__main__":
    test_processed_data_integrity()
