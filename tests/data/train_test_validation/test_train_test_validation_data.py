"""test_train_test_validation_data.py

Module containing the test function for verifying the consistency and integrity
between the training, validation, and testing datasets using the Deepchecks library.

This module ensures that data splits adhere to quality constraints such as dataset
size ratios, feature and label drift limits, leakage prevention, and correlation
consistency.

Functions:
    test_train_test_validation_data() -> None
        Runs a Deepchecks Suite comparing the training set against both the
        validation and testing sets based on configurations.
"""

import pytest
from deepchecks import Suite
from deepchecks.tabular.checks import (
    DatasetsSizeComparison,
    FeatureDrift,
    FeatureLabelCorrelationChange,
    IndexTrainTestLeakage,
    LabelDrift,
    MultivariateDrift,
)

from src.const import DATA_DYNAMIC_MODE, DATA_STATIC_MODE
from tests.const import (
    DATA_TRAIN_TEST_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    DATA_TRAIN_TEST_TESTS_REAL_DATA_RESULTS_SAVE_PATH,
    DATA_TRAIN_TEST_TESTS_STATIC_DATA_RESULTS_SAVE_PATH,
    DATA_TRAIN_TEST_VALIDATION_TESTS_ADD_INDEX_COLUMN,
    DATA_TRAIN_TEST_VALIDATION_TESTS_REMOVE_SEQ_LEN,
    DATA_TRAIN_TEST_VALIDATION_TESTS_SUITE_NAME,
    DATA_TRAIN_VALIDATION_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    DATA_TRAIN_VALIDATION_TESTS_REAL_DATA_RESULTS_SAVE_PATH,
    DATA_TRAIN_VALIDATION_TESTS_STATIC_DATA_RESULTS_SAVE_PATH,
)
from tests.helpers.dc_helpers import initialize_dc_tests, run_dc_suite


@pytest.mark.data_train_test_validation
@pytest.mark.validation
def test_train_test_validation_data() -> None:
    """Runs the Deepchecks test suite comparing training vs.
    validation and training vs. testing sets.

    This test verifies that the dataset splits maintain consistency and
    lack leakage by ensuring adherence to predefined configurations for
    checks, including:
    - DatasetsSizeComparison: Checks that the size ratio between splits meets
                              expectations.
    - IndexTrainTestLeakage: Ensures no index leakage between splits.
    - FeatureLabelCorrelationChange: Assesses stability of feature-label correlation
                                     across splits.
    - MultivariateDrift: Measures overall data distribution drift between splits.
    - LabelDrift: Measures changes in the target variable distribution.
    - FeatureDrift: Measures changes in individual feature distributions.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    # Initialization
    (
        *_,
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        pipeline_config,
        tests_config,
    ) = initialize_dc_tests(
        DATA_TRAIN_TEST_VALIDATION_TESTS_ADD_INDEX_COLUMN,
        DATA_TRAIN_TEST_VALIDATION_TESTS_REMOVE_SEQ_LEN,
    )

    # ----------------------------
    # Suite building
    # ----------------------------
    # Create a suite of tests for training,
    # validation, and testing sets
    suite = Suite(
        DATA_TRAIN_TEST_VALIDATION_TESTS_SUITE_NAME,
        DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(
            ratio=tests_config.data.train_test_validation.ratio.min,
        ),
        IndexTrainTestLeakage().add_condition_ratio_less_or_equal(
            max_ratio=tests_config.data.train_test_validation.index_leakage.max,
        ),
        FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than(
            threshold=tests_config.data.train_test_validation.feat_label_corr_change.max,
        ),
        MultivariateDrift().add_condition_overall_drift_value_less_than(
            max_drift_value=tests_config.data.train_test_validation.multivariate_drift.max,
        ),
        LabelDrift().add_condition_drift_score_less_than(
            max_allowed_drift_score=tests_config.data.train_test_validation.label_drift.max,
        ),
        FeatureDrift().add_condition_drift_score_less_than(
            max_allowed_numeric_score=tests_config.data.train_test_validation.feat_drift.max,
            allowed_num_features_exceeding_threshold=tests_config.data.train_test_validation.feat_drift.max_exceeding_feat,
        ),
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Determine save paths based on the
    # data mode
    if pipeline_config.data.general.mode == DATA_STATIC_MODE:
        train_validation_save_path = (
            DATA_TRAIN_VALIDATION_TESTS_STATIC_DATA_RESULTS_SAVE_PATH
        )
        train_test_save_path = (
            DATA_TRAIN_TEST_TESTS_STATIC_DATA_RESULTS_SAVE_PATH
        )
    elif pipeline_config.data.general.mode == DATA_DYNAMIC_MODE:
        train_validation_save_path = (
            DATA_TRAIN_VALIDATION_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH
        )
        train_test_save_path = (
            DATA_TRAIN_TEST_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH
        )
    else:
        train_validation_save_path = (
            DATA_TRAIN_VALIDATION_TESTS_REAL_DATA_RESULTS_SAVE_PATH
        )
        train_test_save_path = (
            DATA_TRAIN_TEST_TESTS_REAL_DATA_RESULTS_SAVE_PATH
        )

    # Run the suite over training vs.
    # validation sets and training vs.
    # testing sets
    run_dc_suite(
        dc_training_set,
        suite,
        str(train_validation_save_path),
        dc_validation_set,
    )
    run_dc_suite(
        dc_training_set,
        suite,
        str(train_test_save_path),
        dc_testing_set,
    )


if __name__ == "__main__":
    test_train_test_validation_data()
