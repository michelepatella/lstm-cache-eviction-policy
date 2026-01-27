"""test_model_performance.py

Module containing the test function for verifying the performance
of the predictions of the trained model using the Deepchecks library.

This module ensures that the model predictions meets quality constraints defined
in the configuration.

Functions:
    test_model_performance() -> None
        Runs a Deepchecks Suite on the model predictions using the training and
        testing sets based on configuration thresholds.
"""

import pytest
from deepchecks import Suite
from deepchecks.tabular.checks import (
    ConfusionMatrixReport,
    SimpleModelComparison,
    TrainTestPerformance,
)

from const import (
    DATA_DYNAMIC_MODE,
    DATA_STATIC_MODE,
)
from tests.const import (
    MODEL_PERFORMANCE_TESTS_ADD_INDEX_COLUMN,
    MODEL_PERFORMANCE_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    MODEL_PERFORMANCE_TESTS_REAL_DATA_RESULTS_SAVE_PATH,
    MODEL_PERFORMANCE_TESTS_REMOVE_SEQ_LEN,
    MODEL_PERFORMANCE_TESTS_STATIC_DATA_RESULTS_SAVE_PATH,
    MODEL_PERFORMANCE_TESTS_SUITE_NAME,
)
from tests.helpers.dc_helpers import (
    compute_dc_model_predictions,
    initialize_dc_tests,
    run_dc_suite,
)


@pytest.mark.slow
@pytest.mark.model_performance
@pytest.mark.after_model_training
def test_model_performance() -> None:
    """Runs the Deepchecks test suite against the trained model.

    This test tests the model performance by checking its predictions
    by ensuring it adheres to predefined configuration thresholds for
    various checks, including:
        - SimpleModelComparison
        - TrainTestPerformance (Degradation, Min Score)
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
        training_set,
        _,
        testing_set,
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        pipeline_config,
        tests_config,
    ) = initialize_dc_tests(
        MODEL_PERFORMANCE_TESTS_ADD_INDEX_COLUMN,
        MODEL_PERFORMANCE_TESTS_REMOVE_SEQ_LEN,
    )

    # ----------------------------
    # Suite building
    # ----------------------------
    # Create a suite of model tests
    suite = Suite(
        MODEL_PERFORMANCE_TESTS_SUITE_NAME,
        SimpleModelComparison(
            strategy=tests_config.model.performance.baseline_comp.strategy,
            scorers=tests_config.model.performance.baseline_comp.scorers,
        ).add_condition_gain_greater_than(
            min_allowed_gain=tests_config.model.performance.baseline_comp.min_gain,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.performance.train_test.scorers,
        ).add_condition_test_performance_greater_than(
            min_score=tests_config.model.performance.train_test.test_min_score,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.performance.train_test.scorers,
        ).add_condition_train_test_relative_degradation_less_than(
            threshold=tests_config.model.performance.train_test.max_degradation,
        ),
        ConfusionMatrixReport().add_condition_misclassified_samples_lower_than_condition(
            misclassified_samples_threshold=tests_config.model.performance.conf_matrix.max_misclass,
        ),
    )

    # ----------------------------
    # Inference
    # ----------------------------
    # Get both train and test predictions
    # and probabilities to run DeepChecks tests
    # suite on
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = (
        compute_dc_model_predictions(
            training_set,
            testing_set,
            pipeline_config,
        )
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Determine save path based on the
    # data mode
    if pipeline_config.data.general.mode == DATA_STATIC_MODE:
        save_path = MODEL_PERFORMANCE_TESTS_STATIC_DATA_RESULTS_SAVE_PATH
    elif pipeline_config.data.general.mode == DATA_DYNAMIC_MODE:
        save_path = MODEL_PERFORMANCE_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH
    else:
        save_path = MODEL_PERFORMANCE_TESTS_REAL_DATA_RESULTS_SAVE_PATH

    # Run the suite over the training and
    # testing sets as well as the model
    run_dc_suite(
        dc_training_set,
        suite,
        str(save_path),
        dc_testing_set=dc_testing_set,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_proba_train=y_proba_train,
        y_proba_test=y_proba_test,
    )


if __name__ == "__main__":
    test_model_performance()
