"""test_train_test_validation.py

Module containing the test function for verifying the consistency and integrity
between the training, validation, and testing datasets using the Deepchecks library.

This module ensures that data splits adhere to quality constraints such as dataset
size ratios, feature and label drift limits, leakage prevention, and correlation
consistency.

Functions:
    test_train_test_validation() -> None
        Runs a Deepchecks Suite comparing the training set against both the
        validation and testing sets based on configurations.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    DatasetsSizeComparison,
    FeatureDrift,
    FeatureLabelCorrelationChange,
    IndexTrainTestLeakage,
    LabelDrift,
    MultivariateDrift,
    NewLabelTrainTest,
)

from tests.const import (
    TRAIN_TEST_VALIDATION_TESTS_SUITE_NAME,
    TRAIN_VALIDATION_TESTS_RESULTS_SAVE_PATH,
    TRAIN_TEST_TESTS_RESULTS_SAVE_PATH,
)
from tests.helpers import run_dc_suite
from tests.train_test_validation.helpers import (
    initialize_train_test_validation_tests,
)


def test_train_test_validation() -> None:
    """Runs the Deepchecks test suite comparing training vs.
    validation and training vs. testing sets.

    This test verifies that the dataset splits maintain consistency and
    lack leakage by ensuring adherence to predefined configurations for
    checks, including:
    - NewLabelTrainTest: Ensures minimal appearance of new labels in
                         validation/test sets.
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
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        tests_config,
    ) = initialize_train_test_validation_tests()

    # ----------------------------
    # Suite building
    # ----------------------------
    # Create a suite of tests for training,
    # validation, and testing sets
    suite = Suite(
        TRAIN_TEST_VALIDATION_TESTS_SUITE_NAME,
        NewLabelTrainTest(
            random_state=tests_config.seed.value,
        ).add_condition_new_label_ratio_less_or_equal(
            tests_config.train_test_validation.new_label.max_ratio,
        ),
        DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(
            tests_config.train_test_validation.datasets_size_comparison.ratio,
        ),
        IndexTrainTestLeakage(
            random_state=tests_config.seed.value,
        ).add_condition_ratio_less_or_equal(
            tests_config.train_test_validation.index_leakage.max_ratio,
        ),
        FeatureLabelCorrelationChange(
            random_state=tests_config.seed.value,
        ).add_condition_feature_pps_difference_less_than(
            tests_config.train_test_validation.feature_label_correlation_change.threshold,
        ),
        MultivariateDrift(
            random_state=tests_config.seed.value,
        ).add_condition_overall_drift_value_less_than(
            tests_config.train_test_validation.multivariate_drift.max_drift_value,
        ),
        LabelDrift(
            random_state=tests_config.seed.value,
        ).add_condition_drift_score_less_than(
            tests_config.train_test_validation.label_drift.max_allowed_drift_score,
        ),
        FeatureDrift(
            random_state=tests_config.seed.value,
        ).add_condition_drift_score_less_than(
            tests_config.train_test_validation.feature_drift.max_allowed_numeric_score,
            tests_config.train_test_validation.feature_drift.allowed_num_features_exceeding_threshold,
        ),
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Run the suite over training vs.
    # validation sets and training vs.
    # testing sets
    run_dc_suite(
        dc_training_set,
        suite,
        str(TRAIN_VALIDATION_TESTS_RESULTS_SAVE_PATH),
        dc_validation_set,
    )
    run_dc_suite(
        dc_training_set,
        suite,
        str(TRAIN_TEST_TESTS_RESULTS_SAVE_PATH),
        dc_testing_set,
    )


if __name__ == "__main__":
    test_train_test_validation()
