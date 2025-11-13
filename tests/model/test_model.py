"""test_model.py

Module containing the test function for verifying the performance and robustness
of the trained model using the Deepchecks library.

This module ensures that the model meets critical quality constraints defined
in the configuration, covering aspects like performance thresholds, bias checks,
drift stability, inference time, and comparison against a simple baseline.

Functions:
    test_model() -> None
        Runs a Deepchecks Suite on the model using the training and testing sets
        based on configuration thresholds.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    UnusedFeatures,
    SimpleModelComparison,
    WeakSegmentsPerformance,
    ModelInferenceTime,
    SingleDatasetPerformance,
    TrainTestPerformance,
    RocReport,
    ConfusionMatrixReport,
)
from deepchecks.tabular.checks.model_evaluation import PerformanceBias

from const import DATASET_COLUMN_REQUEST_NAME
from tests.const import (
    MODEL_TESTS_SUITE_NAME,
    MODEL_TESTS_RESULTS_SAVE_PATH,
    MODEL_TESTS_ADD_INDEX_COLUMN,
)
from tests.helpers.dc_helpers import run_dc_suite, initialize_dc_tests


def test_model() -> None:
    """Runs the Deepchecks test suite against the trained model.

    This test verifies the quality and stability of the model by
    ensuring it adheres to predefined configuration thresholds for
    various checks, including:
        - UnusedFeatures
        - PerformanceBias
        - SimpleModelComparison
        - WeakSegmentsPerformance
        - ModelInferenceTime
        - SingleDatasetPerformance
        - TrainTestPerformance (Degradation, Imbalance, Min Score)
        - RocReport (AUC)
        - ConfusionMatrixReport
    The suite is run over the training and testing sets along with the model.

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
        model,
        tests_config,
    ) = initialize_dc_tests(MODEL_TESTS_ADD_INDEX_COLUMN)

    # ----------------------------
    # Suite building
    # ----------------------------
    # Create a suite of model tests
    suite = Suite(
        MODEL_TESTS_SUITE_NAME,
        UnusedFeatures(
            random_state=tests_config.seed.value
        ).add_condition_number_of_high_variance_unused_features_less_or_equal(
            max_high_variance_unused_features=tests_config.model.unused_features.max_high_variance_unused_features
        ),
        PerformanceBias(
            protected_feature=DATASET_COLUMN_REQUEST_NAME,
            scorer=tests_config.model.performance_bias.scorer,
            random_state=tests_config.seed.value,
        ).add_condition_bounded_performance_difference(
            lower_bound=tests_config.model.performance_bias.lower_bound
        ),
        SimpleModelComparison(
            strategy=tests_config.model.simple_model_comparison.strategy,
            scorers=tests_config.model.simple_model_comparison.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_gain_greater_than(
            min_allowed_gain=tests_config.model.simple_model_comparison.min_allowed_gain
        ),
        WeakSegmentsPerformance(
            alternative_scorer=tests_config.model.weak_segments_performance.alternative_scorer,
            random_state=tests_config.seed.value,
        ).add_condition_segments_relative_performance_greater_than(
            max_ratio_change=tests_config.model.weak_segments_performance.max_ratio_change
        ),
        ModelInferenceTime(
            random_state=tests_config.seed.value
        ).add_condition_inference_time_less_than(
            value=tests_config.model.inference_time.threshold
        ),
        SingleDatasetPerformance(
            scorers=tests_config.model.single_dataset_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_greater_than(
            threshold=tests_config.model.single_dataset_performance.threshold,
            metrics=tests_config.model.single_dataset_performance.scorers,
            class_mode=tests_config.model.single_dataset_performance.class_mode,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.train_test_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_test_performance_greater_than(
            min_score=tests_config.model.train_test_performance.test_min_score
        ),
        TrainTestPerformance(
            scorers=tests_config.model.train_test_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_train_test_relative_degradation_less_than(
            threshold=tests_config.model.train_test_performance.degradation_threshold,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.train_test_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_class_performance_imbalance_ratio_less_than(
            score=tests_config.model.train_test_performance.score,
            threshold=tests_config.model.train_test_performance.imbalance_threshold,
        ),
        RocReport(
            random_state=tests_config.seed.value
        ).add_condition_auc_greater_than(
            min_auc=tests_config.model.roc_report.min_auc
        ),
        ConfusionMatrixReport(
            random_state=tests_config.seed.value
        ).add_condition_misclassified_samples_lower_than_condition(
            misclassified_samples_threshold=tests_config.model.confusion_matrix_report.misclassified_samples_threshold,
        ),
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Run the suite over the training and
    # testing sets as well as the model
    run_dc_suite(
        dc_training_set,
        suite,
        str(MODEL_TESTS_RESULTS_SAVE_PATH),
        dc_testing_set=dc_testing_set,
        model=model,
    )


if __name__ == "__main__":
    test_model()
